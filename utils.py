# eval_arith.py
#
# Copyright 2009, 2011 Paul McGuire
#
# Expansion on the pyparsing example simpleArith.py, to include evaluation
# of the parsed Elements.
#
# Added support for exponentiation, using right-to-left evaluation of
# operands
#

import string
import cmd
import math
import random
import re
from collections import OrderedDict

import maya.cmds as mc


from pyparsing import Regex, SkipTo, Forward, Word, Combine, Literal, Optional, Group, ParseResults, ParserElement
from pyparsing import nums, alphas, alphanums, oneOf, opAssoc, infixNotation, delimitedList

ParserElement.enablePackrat() # speeds up parser



CONDITION_OPERATORS = {'==': 0, '!=': 1, '>': 2, '>=': 3, '<': 4, '<=': 5}
BUILTIN_VARIABLES   = {'e':   math.e,
                       'pi':  math.pi,
                       'tau': (math.pi * 2),
                       'phi': ((1 + 5 ** 0.5) / 2)}



def parsedcommand(obj):
    """ 
    Decorator to flag documented expression commands available to users.
    Used with the usage() method. 
    """
    obj.parsedcommand = True
    return obj


class Expression(object):
    
    
    # ---------------------------- PARSER DEFINITION ----------------------------- #
    
    def __init__(self, 
                 container=None, 
                 variables=None,
                 debug=False):
        
        self.debug       = debug  # turns on debug prints
        self.container   = None   # name of the container to create and add all created nodes to
        self.constants   = {}     # dict of constants for replacement lookup
        self.constant    = None   # name of network node used to pack constants
        self.repack      = True
        self.attributes  = {}     # dict of attribute names and arguments added to the container
        self.nodes       = []     # keep track of nodes produced by the parser
        self.expression  = []     # the evaluated expression
                                  
        self.private     = {}     # private variables (declared inside the expression)
        self.variables   = {}     # user defined variables (always expanded, will never be assigned)

        expression = Forward()
        integer    = Word(nums) # 42
        real       = (Combine(Word(nums) + 
                              Optional('.' + Word(nums)) +
                              oneOf('e') + 
                              Optional(oneOf('+ -')) + 
                              Word(nums)) |
                      Combine(Optional(Word(nums)) + '.' + Optional(Word(nums)))
                      ) # 3.1416, 2.5e+5
        
                             
        variable  = Combine( '$' + 
                             Word(alphanums) + 
                             Optional('[' + Optional('-') +
                                            Optional(Word(nums)) +
                                            Optional(':') +
                                            Optional('-') +
                                            Optional(Word(nums)) + ']') + 
                             Optional('.'+Word(alphanums) ) 
                             ) # $pi, $list_of_nodes[:-2].t, $MESH.tx                             
                             
                             
        None_var  = Word('None')
        function  = Group(Word(alphas) +
                          Literal('(').suppress() +
                          Group(Optional(delimitedList(expression))) +
                          Literal(')').suppress() 
                          ) # floor(...)
        
                      
        condition_operators = oneOf(' '.join(CONDITION_OPERATORS))
                      
        
        node_attr = Combine( Word(alphanums + '_:') + '.' + Word(alphanums + '_[].') ) # pCube1.tx
        operand   = function | real | integer | node_attr | variable | None_var | condition_operators
        
        # use parse actions to attach EvalXXX constructors to sub-expressions
        operand.setParseAction(self._evalElementOp)
        
        signop = oneOf('+ -')
        multop = oneOf('* / % //')
        plusop = oneOf('+ -')
        expop = Literal('**')        
        
        expression << infixNotation(
            operand,
            [
                (signop, 1, opAssoc.RIGHT, self._evalSignOp),
                (expop,  2, opAssoc.LEFT,  self._evalPowerOp),
                (multop, 2, opAssoc.LEFT,  self._evalMultOp),
                (plusop, 2, opAssoc.LEFT,  self._evalAddOp),
            ],
        )
        

        self.parser = infixNotation(
            expression, [(oneOf('='), 2, opAssoc.LEFT, self._evalAssignOp)]
        )
        
        
        
        # Set the variables if specified
        self.setVariables(variables)
        
        # Set the container if need specifiec
        self.setContainer(container)



    def __call__(self, expression, name=None, variables=None, container=None):
        """
        Evaluates the stored expression and generates the node graph.
    
        Examples
        --------
        >>> from rigging.utils.expression import Expression
        >>> exp = Expression()
        >>> exp('pCube1.t = pCube2.t ** 2')
        """
        self.setExpression(expression)
        self.setVariables(variables)
        self.setContainer(container)
        
        return self.eval()
        
        
    #def __str__(self):
        #return self.expression
        
        
    #def __repr__(self):
        #return self.expression
        
    #def __repr__(self):
        #return type(self).__name__ + '("' + '\n'.join(self.expression) + '")'


    def setContainer(self, container=None):
        """ Creates a container """

        # create the container
        if not self.debug and container:
            self.container = mc.createNode('container', n=container, ss=True)
            mc.setAttr('%s.isCollapsed'%self.container,1)
            
        # add self.nodes to it
        if not self.debug and self.container and self.nodes:
            for node in self.nodes:
                try:
                    mc.container(self.container, edit=True, addNode=node)
                except:
                    pass

        
        
    def setExpression(self, expression=None):
        """ Appends to the expression """

        # store the expression
        if expression:
            self.expression.append(expression)
            
            
    def setVariables(self, variables=None):
        """ Sets user variables """
        
        if variables:
            for var in variables:
                self.variables[var] = variables[var]
                
                
    def getNodes(self):
        """ Returns the expression's nodes """
        
        # if debug mode, or container exists, just return the nodes
        if self.debug or not self.container:
            return self.nodes
        
        return [self.container]
      

    def eval(self):
        """
        eval()
        
            Evaluates the expression and generates its
            represented node tree.
        
            Examples
            --------
            >>> from rigging.utils.expression import Expression
            >>> exp = Expression()
            >>> exp.setExpression('pCube1.t = pCube2.t * 2')
            >>> exp.eval()
        """    
        
        # test to make sure expression brackets are valid
        expression = self.expression[-1]
        opened = expression.count('(')
        closed = expression.count(')')
        if opened != closed:
            raise Exception('Invalid bracket count:  opened [%s], closed [%s]'%(opened, closed))
        
        
        # split the expression into separate line and 
        # evaluate each separately
        if not isinstance(expression, (list, tuple)):
            expression = re.split('\n|;', str(expression))
            

        solution = ''
        for line in expression:
            
            # for evaluation, ignore lines that are commented out.
            # also filter out or trailing comments and semicolons.
            line = line.strip()
            if line and not line.startswith('#'):
    
                if '#' in line:
                    line = line.split('#')[0]
    
                if line.endswith(';'):
                    line = line[:-1]
                    
                
                if self.debug:
                    print ('\nEVALUATING: %s'%line)   
                    
                        
                # Test for assigned variables or piped delimited lists
                stored, line = self._findVariableAssignment(line)
                piped = None
                if not stored:
                    piped, line = self._findDelimitedListAssignment(line)
                

                # Patch fix for human readable condition formatting            
                # cond(a>=b,t,f) --> cond(a,>=,b,t,f)
                regex = r"([^,])(>=|<=|!=|==|>|<)([^,])"
                replace = r"\1,\2,\3"
                line = re.sub(regex, replace, line)
                
                # Patch fix for letting users use "if" instead of "cond"
                regex = re.compile('\W?\w+[(]', re.S)
                line = regex.sub(lambda m: m.group().replace('if(',"cond(",1), line)        
                
                
                # evaluate
                solution = self.parser.parseString(line).asList()


                # process stored variable
                if stored:
                    if self.debug:
                        print ('storing:    %s ---> $%s'%(solution, stored))                         
                    
                    self.private[stored] = solution
                    
                    
                # connect delimited list
                elif piped:
                    for item in piped:

                        self._connectAttr(solution[0], item, align_plugs=True)           
                        #self._evalAssignOp([item,'=',solution[0]])
                        

        # if a container is defined, add all the nodes created underneath
        # and put the expression under it's notes for future reading
        if self.container and self.nodes:
            mc.container(self.container, edit=True, addNode=self.nodes)

        return solution
    
    
                
    def usage(self, query=None, verbose=False):
        """
        usage()
        
            Prints the docstring of every method decorated with @parsedcommand.
        
            Examples
            --------
            >>> from rigging.utils.expression import Parser
            >>> exp = Parser()
            >>> exp.usage()
        """
        
        def isParsedCmd(obj):
            return hasattr(getattr(self, obj), 'parsedcommand')
        
        
        if query in dir(self) and isParsedCmd(query):
            print (getattr(self, query).__doc__ )
    
        else:
            if verbose:
                for x in sorted(dir(self)):
                    if not x.startswith('_') and isParsedCmd(x):
                        print (getattr(self,x).__doc__)
    
            else:
                commands = [x for x in dir(self) if not x.startswith('_') and isParsedCmd(x)]
                cli = cmd.Cmd()
                cli.columnize(sorted(commands), displaywidth=80)    
        
        
    # ----------------------------- PARSER OPERATORS ----------------------------- #

    #def _flatten_lists(self, sequence):
        #""" Flattens a deeply nested list, which can orrur when expanding variables """
        #if not sequence:
            #return sequence
        
        #if isinstance(sequence[0], (list, tuple)):
            #return self._flatten_lists(sequence[0]) + self._flatten_lists(sequence[1:])
        
        #return sequence[:1] + self._flatten_lists(sequence[1:])        


    def _operatorOperands(self, tokenlist):
        """ Generator to extract operators and operands in pairs. """
    
        it = iter(tokenlist)
        while 1:
            try:
                yield (next(it), next(it))
            except StopIteration:
                break
            
        
    def _evalAddOp(self, tokens):
        """ Evaluate addition and subtraction expressions. """
        
        summation = tokens[0][0]
        for op, val in self._operatorOperands(tokens[0][1:]):  
            if op == "+":
                summation = self.add([summation, val])
                
            if op == "-":
                summation = self.sub([summation, val])
                
        return summation
    
    
    def _evalMultOp(self, tokens):
        """ Evaluate multiplication and division expressions. """
        
        prod = tokens[0][0]
        for op, val in self._operatorOperands(tokens[0][1:]): 
            if op == "*":
                prod = self.mult([prod, val])
                
            elif op == "/":
                prod = self.div([prod, val])
                
            elif op == "%":
                prod = self._multiplyDivide('%', [prod, val])
                
            elif op == "//":
                prod = self._multiplyDivide('//', [prod, val])        
                
        return prod   
    
    
    def _evalPowerOp(self, tokens):
        """ Evaluate exponent expressions. """
        
        res = tokens[0][-1]
        for val in tokens[0][-3::-2]:
            res = self.power([val, res])
            
        return res
    
    
    def _evalSignOp(self, tokens):
        """ Evaluate expressions with a leading + or - sign """
        
        res = tokens[0][1]
        
        if self.repack:
            if self.constants:
                inv_map = {v: k for k, v in self.constants.iteritems()}
                
                # is this a mapped constant?
                if res in inv_map:

                    # is it already connected to something?
                    if not mc.listConnections(res):

                        # if not, this is a first instance of an _evalSignOp
                        # we can hack it and skip the multiplication step
                        val = mc.getAttr(res)
                        neg = -1 * val                        
                        
                        # is the negated value already in memory?
                        if neg in self.constants:
                            
                            # unfortunately this will leave a gap in the multi array
                            return self.constants[neg]
                        
                        # replace the value in self.constants
                        mc.setAttr(res, neg) # swap + for -
                        self.constants[neg] = self.constants.pop(val)
                    
                        return self.constants[neg]
                    
        if tokens[0][0] == '-':
            res = self.mult([self._long(-1), res])
            
        return res
    
    
    def _evalAssignOp(self, tokens):
        """ Used for expression variable assignment and connection to nodes. """
        
        dst = tokens[0][:-2]
        src = tokens[0][-1]
        op  = tokens[0][-2]         
        
        # if destination is a tuple, process this as a connection list
        if not isinstance(dst, (tuple, list, set)):
            dst = [dst]
            
        ## if source is matrix type: decompose to matrix ---> transform 
        ## if source is quat: decompose to quat ---> transform
        #decomposeMatrix = False
        #quatToEuler = False

        #if self._isMatrixAttr(src):
            #decomposeMatrix = True
            
        #elif self._isQuatAttr(src):
            #quatToEuler = True
        
        
        # for each destination token
        for item in dst:
            
            # If assignment operator
            if op == '=':
                self._connectAttr(src, item, align_plugs=True)
                
                ## if source is classic float, int, vector plug
                #if not decomposeMatrix and not quatToEuler:
                    #self._connectAttr(src, item, align_plugs=True)
                
                ## if source is matrix or quat type
                #else:
                    
                    ## Matrix to Matrix
                    #if self._isMatrixAttr(item):
                        #pass
                    
                    ## Matrix to Quat
                    #elif self._isQuatAttr(item):
                        #pass
                    
                    ## Matrix to Transform
                    #else:
                        #if decomposeMatrix is True:
                            #decomposeMatrix = self._createNode('decomposeMatrix', ss=self.debug)
                            #mc.connectAttr(src, '%s.inputMatrix'%decomposeMatrix)
                            
                        #_, att = item.split('.')
                        
                        
                        ## scale
                        #if att in ['scale',
                                   #'scaleX',
                                   #'scaleY',
                                   #'scaleZ',
                                   #'s',
                                   #'sx',
                                   #'sy',
                                   #'sz']:
                                
                            #self._connectAttr('%s.outputScale'%decomposeMatrix, item)
                            
                        ## rotate
                        #elif att in ['rotate',
                                     #'rotateX',
                                     #'rotateY',
                                     #'rotateZ',
                                     #'r',
                                     #'rx',
                                     #'ry',
                                     #'rz']:
                                
                            #self._connectAttr('%s.outputRotate'%decomposeMatrix, item)                    
                        
                        ## translate
                        #elif att in ['translate',
                                     #'translateX',
                                     #'translateY',
                                     #'translateZ',
                                     #'t',
                                     #'tx',
                                     #'ty',
                                     #'tz']:
                                
                            #self._connectAttr('%s.outputTranslate'%decomposeMatrix, item)
                            
                        ## shear  
                        #elif att in ['shear']:
                            #self._connectAttr('%s.outputShear'%decomposeMatrix, item)
                            
                            
                        #else:
                            #raise Exception('Matrix object cannot be connected to: %s'%item)
    
        return src
    
    
    # TODO
    # Element evaluation can be cleaned up, and could support lists properly
    def _evalElementOp(self, tokens):
        """ Evaluates a given expression element. """

        def isfloat(x):
            try:
                return eval('type(%s)'%x) == float
            except:
                return False
        
        def isint(x):
            try:
                return eval('type(%s)'%x) == int
            except:
                return False                  
        
        # merge variables for simplicity.
        # private vars always supercede public.
        variables = self.variables.copy()
        variables.update(self.private)               

        
        # is element None?
        if tokens[0] == 'None':
            return None
        
        # is it one of the CONDITION_OPERATORS?
        elif tokens[0] in CONDITION_OPERATORS:
            return tokens[0]
        
        # is element a function?
        elif isinstance(tokens[0], ParseResults):
            function = tokens[0][0]
            return getattr(self, function)(tokens[0][1])        
                         
        # is element a float?  
        elif isfloat(tokens[0]):
            return self._double(float(tokens[0]), constant=True)
                
        # is element an int?
        elif isint(tokens[0]):
            return self._long(int(tokens[0]), constant=True)           
        
        # is element a node.attr
        elif mc.objExists(tokens[0]):
                return tokens[0]
              
        # is element a variable?
        elif variables and tokens[0].startswith('$'): 
            var, index, attr = self._splitVariable(tokens[0])
            
            # is the variable declared?
            values = None
            if var in variables:
                values = variables[var]
                
                if isinstance(values, (list, tuple, ParseResults)):
                    if isinstance(values, ParseResults):
                        values = values.asList()

                    if index is not None:
                        if isinstance(index, slice):
                            values = values[index]
                        else:
                            values = [values[index]]
                else:
                    values = [values]

                if attr:
                    for i,v in enumerate(values):
                        values[i] = '%s.%s'%(v,attr)
                
         
                for i,v in enumerate(values):
                    values[i] = self._evalElementOp([v])
                
                
                if len(values) == 1:
                    if self.debug:
                        print 'expanding:  %s ---> %s'%(tokens[0], values[0])
                    return values[0]
                
                print 'expanding:  %s ---> %s'%(tokens[0], values)
                return values


        # syntax error
        else:
            raise Exception('Invalid token: %s'%tokens[0])   
    
    
        
    ## TODO
    ## Element evaluation can be cleaned up, and could support lists properly
    #def _evalElementOp(self, tokens):
        #""" Evaluates a given expression element. """
        
        #def isfloat(x):
            #try:
                #return eval('type(%s)'%x) == float
            #except:
                #return False
        
        #def isint(x):
            #try:
                #return eval('type(%s)'%x) == int
            #except:
                #return False                  
        
        ## merge variables for simplicity.
        ## private vars always supercede public.
        #variables = self.variables.copy()
        #variables.update(self.private)               

        
        ## is element None?
        #if tokens[0] == 'None':
            #return None
        
        ## is it one of the CONDITION_OPERATORS?
        #elif tokens[0] in CONDITION_OPERATORS:
            #return tokens[0]
        
        ## is element a function?
        #elif isinstance(tokens[0], ParseResults):
            #function = tokens[0][0]
            #return getattr(self, function)(tokens[0][1])        
        
        
        ## is element a variable?
        #elif variables and tokens[0].startswith('$'):
            
            #var, index, attr = self._splitVariable(tokens[0])

            ## is the variable declared?
            #values = None
            #if var in variables:
                #values = variables[var]
                
                ## if value is a list
                #if not isinstance(values, (list, tuple, ParseResults)) and index:
                    #raise Exception('Variable slicing/indexing only allowed on lists and tuples: %s'%tokens[0])
                #else:
                    #if self.debug:
                        #print ('expanding:  %s ---> %s'%(var, values))              


                ## if the value is a float
                #if isfloat(values) and not attr:
                    #node = self._double(values)
                    #self.private[var] = node
                    #return node
                
                ## if the value is an int
                #elif isint(values) and not attr:
                    #node = self._long(values)
                    #self.private[var] = node
                    #return node                    
                
                ## if value is a string --> node.attr
                #elif isinstance(values, (str, unicode)):
                    #if attr:
                        #return '%s.%s'%(values.split('.')[0], attr)
                    #else:
                        #return values

                
                ## if value is a list
                #elif isinstance(values, (list, tuple, ParseResults)):

                    ## slice?
                    #if index:
                        #values = values[index]
                        #if isinstance(index, int):
                            #print index
                            #return self._evalElementOp([str(values)])
                            
                            
                        #if not isinstance(values, (tuple, list)):
                            #values = [values]

                        #if self.debug:
                            #print ('expanding:  %s ---> %s'%(tokens[0][1:].split('.')[0], values))                           
                    #else:

                        #if self.debug:
                            #print ('expanding:  %s ---> %s'%(var, values))
                            

                    ## if all of the elements are strings
                    #if all([isinstance(x, (str, unicode)) for x in values]) and not any([(isfloat(x) or isint(x)) for x in values]):
                        #if attr:
                            #values = ['%s.%s'%(x.split('.')[0], attr) for x in values]
                        #return values
                    
                    
                        
                    #elif not attr:
                        
                        ## is it a vector?
                        #if len(values) == 3:
                            ##return self._double3(values)
                            #node = self._double3(values)
                            #self.private[var] = node
                            #return node                            

                        
                        ## is it a 4x4 matrix?
                        #elif len(values) == 16:
                            #node = self._createNode('fourByFourMatrix', ss=True)
                            #idx = 0
                            #for i in range(4):
                                #for j in range(4):
                                    #mc.setAttr('%s.in%s%s' % (node, i, j), values[idx])
                                    #idx += 1
                                    
                            ##return '%s.output'%node
                            #self.private[var] = '%s.output'%node    
                            #return '%s.output'%node                       
                            
                    
        ## is element a float?  
        #elif isfloat(tokens[0]):
            #return self._double(float(tokens[0]))
                
        ## is element an int?
        #elif isint(tokens[0]):
            #return self._long(int(tokens[0]))           
        
        ## is element a node.attr
        #elif mc.objExists(tokens[0]):
                #return tokens[0]
            
        ## syntax error
        #else:
            #raise Exception('Invalid token: %s'%tokens[0])
            
  

    
    # -------------------------------- UTILITIES --------------------------------- #
    
    def _nextFreePlug(self, query):
        """ 
        Returns the next free plug index.
        """
    
        # no need to go any further if not a node.attr
        if not '.' in query:
            return query
    
        # find where the multi attr entry point is
        split = query.split('.')
        for i in range(2, len(split) + 1):
            start = '.'.join(split[:i])
            end = '.'.join(split[i:])
            search = None
            try:
                mc.listAttr('%s[0]' % start, m=True)
                search = '{0}[%s].{1}'.format(start, end)
                break
            except:
                pass
    
        # if we don't have a valid entry point, return query
        if search is None:
            return query
    
        # find the next free plug
        index = 0
    
        while True:
            if not mc.listConnections(search % index, s=True, d=False):
                result = search % index
                if result.endswith('.'):
                    return result[:-1]
                return result
    
            index += 1
    
    
    def _listPlugs(self, query):
        """
        Lists a node.attr's plug array attributes by returning the next free plugs.
        """
        # no need to go any further if not a node.attr
        if not '.' in query:
            return [query]
    
        query = self._nextFreePlug(query)
        node  = query.split('.')[0]
    
        return ['%s.%s' % (node, x) for x in mc.listAttr(query)]
    

    def _getPlugs(self, query, compound=True):
        """
        Enumerates input plugs and matches them as sets of same size.
        ex: [pCube2.v, pCube1.t] ---> [[pCube2.v, pCube2.v, pCube2.v], [pCube1.tx, pCube1.ty, pCube1.tz]]
        ex: compound=True  and [pCube1.t, pCube2.t] ---> [[pCube1.t], [pCube2.t]]
        ex: compound=False and [pCube1.t, pCube2.t] ---> [[pCube1.tx, pCube1.ty, pCube1.tz], [pCube2.tx, pCube2.ty, pCube2.tz]]
        """
    
        if not isinstance(query, (list, tuple, ParseResults)):
            query = [query]
    
        attrs = []
        for obj in query:
            attrs.append(self._listPlugs(obj))
            
        counts = [len(x) for x in attrs]

        if counts:
            maxi = max(counts) - 1
    
            # !!! HACK !!! #
            # If one of the inputs is a choice node, we force compound mode
            choice_test = False
            try:
                choice_test = any([mc.nodeType(q) == 'choice' for q in query])
            except:
                pass
    
            # If all counts the same
            if len(query) > 1 and (choice_test or ([counts[0]] * len(counts) == counts and compound)):
                return [[x[0]] for x in attrs]
    
            else:

                # Compound mode off
                if maxi == 0 and not compound:
                    return attrs
    
                result = [[None] * maxi for _ in query]

                for i in range(len(query)):
                    for j in range(maxi):
                        if counts[i] == 1:
                            result[i][j] = attrs[i][0]
                        else:
                            result[i][j] = attrs[i][min(counts[i], j + 1)]

                return result
        


    def _connectAttr(self, src, dst, align_plugs=True):
        # Connection logic
        # - one to one    x   --> x
        # - one to many   x   --> x,X,r,R
        #                 x   --> y,Y,g,G
        #                 x   --> z,Z,b,B
        # - comp to comp  xyz --> xyz 
        #                 matrix ---> matrix
        # - many to many  z   --> x,X,r,R
        #                 z   --> y,Y,g,G
        #                 z   --> z,Z,b,B        
        #
        # (EXPERIMENTAL)
        # WITH ALIGN PLUGS
        # - many to many  x   --> x,X,r,R
        #                 y   --> y,Y,g,G
        #                 z   --> z,Z,b,B

        src, dst = self._getPlugs([src, dst])
        
        # if source is matrix type: decompose to matrix ---> transform 
        # if source is quat: decompose to quat ---> transform
        decomposeMatrix = False
        quatToEuler = False
    
        if self._isMatrixAttr(src[0]):
            decomposeMatrix = True
    
        elif self._isQuatAttr(src[0]):
            quatToEuler = True        
        
        
        if not decomposeMatrix and not quatToEuler:
        
            # match indices according to order
            if align_plugs and len(src) > 1:
    
    
                # try to find where the destination is in the array and 
                # match to source via it's index
                try:
    
                    node = dst[0].split('.')[0]
                    att  = '.'.join(dst[0].split('.')[1:])
                    src_ = []
                    dst_parent = mc.attributeQuery(att, node=node, lp=True)
    
                    if dst_parent:
                        attrs = self._listPlugs('%s.%s'%(node, dst_parent[0]))[1:]
    
                        if len(attrs) == len(src):
    
                            for i in range(len(src)):
                                idx = attrs.index(dst[i])
                                src_.append(src[idx])
    
                            src = src_
    
                except:
                    pass
    
    
            for i in range(len(src)):
                if self.debug:
                    print ('connecting: %s ---> %s'%(src[i],dst[i]))
    
                mc.connectAttr(src[i], dst[i], f=True)
             
             
        else: 
            
            src = src[0]
            dst = dst[0]
            
            # Matrix to Matrix
            if self._isMatrixAttr(dst):
                mc.connectAttr(src, dst, f=True)
            
            # Quat to Quat
            elif self._isQuatAttr(dst):
                mc.connectAttr(src, dst, f=True)
            
            # Matrix to Transform
            else:
                if decomposeMatrix is True:
                    decomposeMatrix = self._createNode('decomposeMatrix', ss=self.debug)
                    mc.connectAttr(src, '%s.inputMatrix'%decomposeMatrix)
                    
                _, att = dst.split('.')
                
                
                # scale
                if att in ['scale',
                           'scaleX',
                           'scaleY',
                           'scaleZ',
                           's',
                           'sx',
                           'sy',
                           'sz']:
                        
                    self._connectAttr('%s.outputScale'%decomposeMatrix, dst)
                    
                # rotate
                elif att in ['rotate',
                             'rotateX',
                             'rotateY',
                             'rotateZ',
                             'r',
                             'rx',
                             'ry',
                             'rz']:
                        
                    self._connectAttr('%s.outputRotate'%decomposeMatrix, dst)                    
                
                # translate
                elif att in ['translate',
                             'translateX',
                             'translateY',
                             'translateZ',
                             't',
                             'tx',
                             'ty',
                             'tz']:
                        
                    self._connectAttr('%s.outputTranslate'%decomposeMatrix, dst)
                    
                # shear  
                elif att in ['shear']:
                    self._connectAttr('%s.outputShear'%decomposeMatrix, dst)
                    
                    
                else:
                    
                    try:
                        mc.connectAttr(src, dst, f=True)
                    except:
                        raise Exception('Matrix object cannot be connected to: %s'%dst)
            
            
            
                
                



    #def _connectAttr(self, src, dst, align_plugs=True):
        ## Connection logic
        ## - one to one    x   --> x
        ## - one to many   x   --> x,X,r,R
        ##                 x   --> y,Y,g,G
        ##                 x   --> z,Z,b,B
        ## - comp to comp  xyz --> xyz 
        ##                 matrix ---> matrix
        ## - many to many  z   --> x,X,r,R
        ##                 z   --> y,Y,g,G
        ##                 z   --> z,Z,b,B        
        ##
        ## (EXPERIMENTAL)
        ## WITH ALIGN PLUGS
        ## - many to many  x   --> x,X,r,R
        ##                 y   --> y,Y,g,G
        ##                 z   --> z,Z,b,B
        
        #src, dst = self._getPlugs([src, dst])
    
        ## match indices according to order
        #if align_plugs and len(src) > 1:
            

            ## try to find where the destination is in the array and 
            ## match to source via it's index
            #try:
                
                #node = dst[0].split('.')[0]
                #att  = '.'.join(dst[0].split('.')[1:])
                #src_ = []
                #dst_parent = mc.attributeQuery(att, node=node, lp=True)
                
                #if dst_parent:
                    #attrs = self._listPlugs('%s.%s'%(node, dst_parent[0]))[1:]
    
                    #if len(attrs) == len(src):
    
                        #for i in range(len(src)):
                            #idx = attrs.index(dst[i])
                            #src_.append(src[idx])
                    
                        #src = src_
                        
            #except:
                #pass

                
            
        #for i in range(len(src)):
            #if self.debug:
                #print ('connecting: %s ---> %s'%(src[i],dst[i]))
                
            #mc.connectAttr(src[i], dst[i], f=True)    
    
    
    def _createNode(self, *args, **kwargs):
        """ 
        Wrapper around mc.createNode to keep a tally of created nodes under self.nodes
        """
        node = mc.createNode(*args, **kwargs)
        self.nodes.append(node)
        
        if self.container:
            try:
                mc.container(self.container, edit=True, addNode=node)
            except:
                pass                 
        
        return node   
    
    
    def _isLongAttr(self, item):
        """ 
        Check if plug is a type int.
        """
        return mc.getAttr(item) in [int, bool]
    
    
    def _isDoubleAttr(self, item):
        """ 
        Check if plug is a type int, bool or enum (all valid ints).
        """
        return mc.getAttr(item) is float
                    
                    
    def _isLong3Attr(self, item):
        """ 
        Check if plug is a type int.
        """
        return mc.getAttr(item, type=True) == 'long3'
    
    
    def _isDouble3Attr(self, item):
        """ 
        Check if plug is a type int, bool or enum (all valid ints).
        """
        mc.getAttr(item, type=True) == 'double3'          
                    
                    
    def _isMatrixAttr(self, item):
        """ 
        Check if plug is a type matrix or not.
        """
        return mc.getAttr(item, type=True) == 'matrix'

    
    def _isQuatAttr(self, item):
        """ 
        Check if plug is a type quaternion or not.
        """
        return mc.getAttr(item, type=True) in ['double4', 'TdataCompound']



    #def _packNumericConstants(self, line):
        #"""
        #Extracts all numeric constants from the expression and
        #adds them to a single network node. Each constant will
        #be added as a variable for future lookup.
        #"""
        
        #def isfloat(x):
            #try:
                #return eval('type(%s)'%x) == float
            #except:
                #return False

        #constants = [x[0] for x in Regex(r'[+-]?\d+\.?\d*([eE][+-]?\d+)?').searchString(line).asList()] 
        #constants = sorted(set(constants), key=len)[::-1] # reverse order for safe replace
        #constants = [x for x in constants if float(x)]
 
 
        #if constants:
            
            ## create the constant node if does not exist
            #if not self.constant:
                #self.constant = self._createNode('network', name='constants1', ss=True)
                #mc.addAttr(self.constant, ln = "integers", at="long",   m=True)
                #mc.addAttr(self.constant, ln = "floats",   at="double", m=True)              
            
            
            ## step 1, replace any constants with random garbage
            #garbage = {} # 'DFDFAFDFD':'3.14'
            #for x in constants:
                #random_garbage = ''.join(random.choice(string.ascii_uppercase) for _ in range(16))
                #garbage[x] = random_garbage
    
                #line = line.replace(x, random_garbage) 
                
                
            ## step 2, properly replace random garbage with proper constant
            #for x in constants:
                
                #plug = None
                #if x in self.constants:
                    #plug = self.constants[x]
                
                #else:       
                    #if isfloat(x):
                        #plug = self._nextFreePlug('%s.floats'%self.constant)
                        #mc.setAttr(plug, float(x))
                    #else:
                        #plug = self._nextFreePlug('%s.integers'%self.constant)
                        #mc.setAttr(plug, int(x))
                    

                ## replace using the constant node
                #line = line.replace(garbage[x], plug)  


        ## return the line    
        #return line
            

                
    
    
    
    
    
    
    
    def _findVariableAssignment(self, line):
        """
        Extracts assigned variables from a given string.
        Only variables in self.private can be assigned.
        Variables in self.variables will be expanded in a EqualOp situation
        
        Returns: the name of the variable and the line stripped of the declaration
        """
        line = ''.join(line.strip().split())
        variable  = Combine( '$' + 
                             Word(alphanums) + 
                             Optional('[' + Optional('-') +
                                            Optional(Word(nums)) +
                                            Optional(':') +
                                            Optional('-') +
                                            Optional(Word(nums)) + ']') + 
                             Optional('.'+Word(alphanums) ) 
                             ) # $pi, $list_of_nodes[:-2].t, $MESH.tx           
        
        var = Combine(variable+Literal('=')).searchString(line,1).asList()
        if var:
            var = var[0][0]
            start = line.find(var)
            
            if start == 0:
                
                # if there's an index or a .attr in the variable,
                # this is expected to be a user variable and will
                # be expanded by the parser.
                if '[' in var or '.' in var:
                    return None, line
                
                
                # verify var is not a user defined variable (cannot be set!)
                if not var[1:-1] in self.variables:
                    end   = start + len(var)
                    line  = line[end:]
                
                    # return var without '$'
                    return var[1:-1], line
            
        return None, line
    
    
    def _findDelimitedListAssignment(self, line):
        """
        Extracts assigned delimited list from a given string.
        ex: pCube1.t, pCube2.t, pCube3.t = sin(45)

        Returns: the list and the line stripped of the declaration
        """
        
        line = ''.join(line.strip().split())
        
        # confirm the existence of an assigned delimited list
        node_attr = Combine( Word(alphanums + '_:') + '.' + Word(alphanums + '_[].') )
        delim = Group(delimitedList(node_attr, delim=',') + '=')
        delim = delim.searchString(line,1).asList()
        
        if delim:
            
            # find out where it is
            delim_ = SkipTo('=').parseString(line)[0]
            start = line.find(delim_)
            
            if start == 0:
                end = start + len(delim_)+1
                
                # catch possible '==' mistake
                if not line[end] == '=':
                    line  = line[end:]
                    return delim[0][0][:-1], line
        
        return None, line    

    
    
    def _splitVariable(self, var):
        """ 
        Breaks down the variable into (var, slice, attr)
        """
        # cleanup variable of any whitespaces
        var  = ''.join(var.split())
        attr = None
        if var.startswith('$'):
            var = var[1:]
            
        # test for variable override
        if '.' in var:
            split = var.split('.')
            var   = split[0]
            attr  = '.'.join(split[1:])
            
            
        # test if we're slicing an index
        var_slice = None
        if '[' in var:
            var_slice = var[var.find('[')+1:var.find(']')]
            var = var.split('[')[0]
            if ':' in var_slice:
                var_slice = slice(*[{True: lambda n: None, False: int}[x == ''](x) for x in (var_slice.split(':') + ['', '', ''])[:3]])
            else:
                var_slice = int(var_slice)
         
        # if var is blank, this is a $.attr situation, and the variable 
        # is on the container, make sure container exists.
        #if not var and not self.container:
            #raise Exception('Missing container for attribute $.%s'%attr)

        return (var, var_slice, attr)
        
    
    
    

    # ----------------------------- GENERAL NODES ------------------------------ #
    
    # TODO: add input as a time offset
    @parsedcommand
    def frame(self, items):
        """ 
        frame()
        
            Outputs "current frame" via an infinite linear motion curve.
        
            Examples
            --------
            >>> frame() # returns a current time slider value.
        """
    
        if items:
            raise Exception('frame functions does not expect inputs')
    
    
        curve = self._createNode('animCurveTU', n='frame1', ss=True)
        mc.setKeyframe(curve, t=0, v=0.)
        mc.setKeyframe(curve, t=1, v=1.)
        mc.keyTangent(curve, e=True, itt='linear', ott='linear')
        mc.setAttr('%s.preInfinity' % curve, 4)
        mc.setAttr('%s.postInfinity' % curve, 4)
    
        return '%s.o' % curve



    # TODO: this is still experimental.
    @parsedcommand
    def noise(self, items):
        """ 
        noise(<input>)
        
            Creates a pseudo random function via perlin noise.
        
            Examples
            --------
            >>> noise(pCube1.tx) # Applies noise to pCube1's translateX value.
            >>> noise(pCube1.t)  # Applies noise to pCube1's [tx, ty, tz] values.
        """
    
        # Handle single value or vector
        items = self._getPlugs(items, compound=False)
        
        exp = Expression(container='noise1', debug=self.debug)
        results = []
        for i in range(len(items[0])):
            plug = items[0][i]
    
            # create a noise node
            noise = exp._createNode('noise_node1', ss=True)
            mc.setAttr('%s.ratio' % noise, 1)
            mc.setAttr('%s.noiseType' % noise, 4)  # set noise to 4d perlin (spacetime)
            mc.setAttr('%s.frequencyRatio' % noise, BUILTIN_VARIABLES['phi'])  # set freq to golden value
    
            # animate the time function with a random seed value
            v0 = random.randint(-1234, 1234) + random.random()
            v1 = v0 + 1
            mc.setKeyframe(noise, attribute='time', t=0, v=v0)
            mc.setKeyframe(noise, attribute='time', t=1, v=v1)
            
            curve = mc.listConnections('%s.time' % noise)[0]
            exp.nodes.append(curve)
            mc.keyTangent(curve, e=True, itt='linear', ott='linear')
            mc.setAttr('%s.preInfinity' % curve, 4)
            mc.setAttr('%s.postInfinity' % curve, 4)
    
            # clamp output because sometimes noise goes beyond 0-1 range
            results.append(exp('%s * clamp(%s.outColorR, 0.001,0.999)' % (plug, noise))[0])
            #exp = '%s * clamp(%s.outColorR, 0.001,0.999)' % (plug, noise)
            #results.append(self.eval(exp))
    
        result = None
        if len(results) == 1:
            result = results[0]
    
        elif len(results) == 3:
            result = exp._double3()
            for i, xyz in enumerate(['X', 'Y', 'Z']):
                self._connectAttr(results[i], '%s%s' % (result, xyz))
            
        else:
            raise Exception('random functions ony supports 1 or 3 plugs. given: %s'%results)

        self.nodes.extend(exp.getNodes())
        return result



    @parsedcommand
    def lerp(self, items):
        """ 
        lerp(<input>, <input>, <weight>)
        
            Linear interpolation between two inputs and a weight value.
        
            Examples
            --------
            >>> lerp(pCube1.tx, pCube2.tx, pCube3.weight)  # Computes the magnitude of [tx, ty, tz].
        """
        if len(items) != 3:
            raise Exception('lerp requires 3 inputs, given: %s' % items)
    
        exp = Expression(container='lerp1', variables=locals(), debug=self.debug)
        node = exp('$items[0] + $items[2] * ($items[1]-$items[0])', variables=locals())[0]

        self.nodes.extend(exp.getNodes())
        return node
    
    
    @parsedcommand
    def elerp(self, items):
        """ 
        elerp(<input>, <input>, <weight>)
        
            Exponent linear interpolation between two inputs and a weight value.
        
            Examples
            --------
            >>> elerp(pCube1.tx, pCube2.tx, pCube3.weight)  # Computes the magnitude of [tx, ty, tz].
        """
        if len(items) != 3:
            raise Exception('lerp requires 3 inputs, given: %s' % items)
    
    
        exp = Expression(container='elerp1', variables=locals(), debug=self.debug)
        node = exp('$items[0]**rev($items[2]) * $items[1]**$items[2]')[0]
        self.nodes.extend(exp.getNodes())
        return node 
    
    @parsedcommand
    def slerp(self, items):
        """ 
        slerp(<input>, <input>, <weight>)
        
            Spherical linear interpolation between two inputs and a weight value.
        
            Examples
            --------
            >>> slerp(pCube1.tx, pCube2.tx, pCube3.weight)  # Computes the magnitude of [tx, ty, tz].
        """
        if len(items) != 3:
            raise Exception('slerp requires 3 inputs, given: %s' % items)
    
        v0 = items[0]
        v1 = items[1]
        blend = items[2]
    
        exp = Expression(container='slerp1', variables=locals(), debug=self.debug)   
        exp('$angle = acos(dot(unit($v0), unit($v1)))')
        node = exp('(($v0*sin(rev($blend)*$angle)) + ($v1*sin($blend*$angle)))/sin($angle)')
        self.nodes.extend(exp.getNodes())
        return node         
        
        
    


    @parsedcommand
    def mag(self, items):
        """ 
        mag(<input>)
        
            Returns the magnitude of a vector.
        
            Examples
            --------
            >>> mag(pCube1.t)  # Computes the magnitude of [tx, ty, tz].
        """
        if len(items) != 1:
            raise Exception('mag requires 1 input, given: %s' % items)
    
        node = self._createNode('distanceBetween', ss=True)
        self._connectAttr(items[0], '%s.point1' % node)
    
        return '%s.distance' % node


    @parsedcommand
    def cond(self, items):
        """ 
        cond(<input> <op> <input>, <input if true>, <input if false>)
        
            Creates condition node to solve "if" statements.
        
            Examples
            --------
            >>> cond(pCube1.t > pCube2.t, 0, pCube3.t)
            >>> cond(pCube1.rx < 45, pCube1.rx, 45) # outputs pCube1.rx's value with a maximum of 45
        """

        if len(items) != 5:
            raise Exception('cond() needs 5 items: [a,cond_op,b,val if true,val if false]. Given: %s' % items)
    
        A, op, B, true, false = self._getPlugs(items)

        
        if op[0] not in CONDITION_OPERATORS:
            raise Exception('unsupported condition operator. given: %s'%op[0])
    
        if len(A) == 1 and len(B) == 1:
            node = self._createNode('condition', ss=True)
    
            self._connectAttr(A[0], '%s.firstTerm' % node)
            mc.setAttr('%s.operation' % node, CONDITION_OPERATORS[op[0]])
            self._connectAttr(B[0], '%s.secondTerm' % node)
            self._connectAttr(true[0], '%s.colorIfTrue' % node)
            self._connectAttr(false[0], '%s.colorIfFalse' % node)
    
            return '%s.outColorR' % node
    
    
        elif len(A[0]) > 1:
            vec = self._double3()
            xyz = self._listPlugs(vec)[1:]
            for i in range(len(A)):
                self._connectAttr(self.cond([A[i], op[i], B[i], true[i], false[i]]), xyz[i])
    
            return vec  
        
        
        
        
    # TODO: Use None to only limit one way, like clamp(pCube1.ty, 0, None)
    @parsedcommand
    def clamp(self, items):
        """ 
        clamp(<input>, <input min>, <input max>)
        
            Clamps values between a min and a max.
        
            Examples
            --------
            >>> clamp(pCube1.ty, 0, pCube2.ty) # clamps pCube1.ty value between 0 and pCube2.ty
            >>> clamp(pCube1.t, -1, 1) # clamps [tx, ty, tz] of pCube1 between -1 and 1
        """
    
        if len(items) != 3:
            raise Exception('clamp() requires 3 inputs, given: %s' % items)
    
        # all inputs differ, use a clamp node
        if items[0] != items[1] and items[0] != items[2]:
            node = self._createNode('clamp', ss=True)
            mc.setAttr('%s.renderPassMode' % node, 0)
            self._connectAttr(items[1], '%s.min' % node)
            self._connectAttr(items[2], '%s.max' % node)
            self._connectAttr(items[0], '%s.input' % node)
    
            counts = self._getPlugs(items, compound=False)
            if all(len(x) == 1 for x in counts):
                return '%s.outputR' % node
    
            return '%s.output' % node
    
        # shared input, use 2 conditions
        # max(min(my_value, max_value), min_value)
        MIN = self.cond([items[0], '<', items[2], items[0], items[2]])
        return self.cond([MIN, '>', items[1], MIN, items[1]])       
    
    
    
    
    # ----------------------------- NUMERICAL NODES ------------------------------ #    
    def _number(self, value=0, at='double', name='number1', constant=False):
        """
        Creates a single numeric value using a network node (default float).
        """
        
        # return repacked constant
        if constant and self.repack:
            
            if value in self.constants:
                return self.constants[value]
            
            if not self.constant:
                self.constant = self._createNode('network', name='constants1', ss=True) 
                
            if not mc.attributeQuery(at, node=self.constant, exists=True):
                mc.addAttr(self.constant, ln=at, at=at, dv=0, m=True)
                
            index = mc.listAttr('%s.%s'%(self.constant,at), m=True) 
            index = len(index) if index is not None else 0
            plug = '%s.%s[%s]'%(self.constant, at, index)
            
            mc.setAttr(plug, value)
            self.constants[value] = plug
            
            return plug
        
        
        node = self._createNode('network', name=name, ss=True)
        mc.addAttr(node, ln='output', at=at, dv=float(value), keyable=True)
        
        return '%s.output' % node
    

    def _number3(self, value=(0,0,0), at='double', name='number1', constant=False):
        """
        Creates a vector numeric value using a network node (default double3).
        """
        
        # return repacked constant
        if constant and self.repack:        
        
            # return prepacked constant
            if value in self.constants:
                return self.constants[value]
            
            att = '%s3'%at
            if not self.constant:
                self.constant = self._createNode('network', name='constants1', ss=True)
                
            if not mc.attributeQuery(att, node=self.constant, exists=True):
                mc.addAttr(self.constant, ln=att, at=att, m=True)        
                mc.addAttr(node, ln='%sX'%at, at=at, dv=0, p=att)
                mc.addAttr(node, ln='%sY'%at, at=at, dv=0, p=att)
                mc.addAttr(node, ln='%sZ'%at, at=at, dv=0, p=att)
                
            index = mc.listAttr('%s.%s'%(self.constant,att), m=True) 
            index = len(index) if index is not None else 0
            plug = '%s.%s[%s]'%(self.constant, att, index)
            mc.setAttr(plug, *value)
            self.constants[value] = plug
            return plug
                
        else:
            
            node = self._createNode('network', name=name, ss=True)
            mc.addAttr(node, ln='output', at='%s3' % at, keyable=True)
            mc.addAttr(node, ln='outputX', at=at, p='output', dv=value[0], keyable=True)
            mc.addAttr(node, ln='outputY', at=at, p='output', dv=value[1], keyable=True)
            mc.addAttr(node, ln='outputZ', at=at, p='output', dv=value[2], keyable=True)
        
            return '%s.output' % node  
        
    
    
    def _long(self, value=0, name='integer1', constant=False):
        """
        Creates a single numeric integer using a network node.
        """        
        return self._number(value=value,
                            name=name,
                            at='long',
                            constant=constant)
    
    
    def _long3(self, value=(0,0,0), name='integer1', constant=False):
        """
        Creates a vector numeric integer using a network node.
        """        
        return self._number3(value=value,
                             name=name,
                             at='long',
                            constant=constant)    
    
    
    def _double(self, value=0.0, name='float1', constant=False):
        """
        Creates a single numeric float using a network node.
        """        
        return self._number(value=value,
                            name=name,
                            at='double',
                            constant=constant)
    
    
    def _double3(self, value=(0.,0.,0.), name='vector1', constant=False):
        """
        Creates a vector numeric float using a network node.
        """        
        return self._number3(value=value,
                            name=name,
                            at='double',
                            constant=constant)    

    
    
    # -------------------------- PEMDAS MATH FUNCTIONS --------------------------- #

    def _multiplyDivide(self, op, items):
        """
        PEMDAS multiply/divide operations
        """
        mat0 = self._isMatrixAttr(items[0])
        mat1 = self._isMatrixAttr(items[1])
    
        if not mat0 and not mat1:
            
            
            # multiply
            if op == '*':
                node = self._createNode('multiplyDivide', ss=True, n='multiply1')
                mc.setAttr('%s.operation' % node, 1)

            # divide
            elif op == '/':
                node = self._createNode('multiplyDivide', ss=True, n='divide1')
                mc.setAttr('%s.operation' % node, 2)

            # power
            elif op == '**':
                node = self._createNode('multiplyDivide', ss=True, n='exponent1')
                mc.setAttr('%s.operation' % node, 3)

            # floor division
            elif op == '//':
                exp = Expression(container='floorDivision1', debug=self.debug)
                result = exp('floor(%s/%s)' % (items[0], items[1]))[0]
                
                self.nodes.extend(exp.getNodes())
                return result                
                #return self.eval('floor(%s/%s)' % (items[0], items[1]))
            
            # modulo
            elif op == '%':
                exp = Expression(container='modulo1' ,debug=self.debug)
                result = exp('%s - floor(%s/%s)*%s' % (items[0], items[0], items[1], items[1]))[0]
                
                self.nodes.extend(exp.getNodes())                
                return result
                #return self.eval('%s - floor(%s/%s)*%s' % (items[0], items[0], items[1], items[1])) 
  
            
            else:
                raise Exception('unsupported operator: %s' % op)
    
            self._connectAttr(items[0], '%s.input1' % node)
            self._connectAttr(items[1], '%s.input2' % node)
    
            # Force single output if both inputs are single numerics
            counts = self._getPlugs(items, compound=False)
            if all(len(x) == 1 for x in counts):
                return '%s.outputX' % node
    
            return '%s.output' % node
    
        else:
    
            # is this a mat * mat
            if mat0 and mat1:
                return self.matrixMult(items)
    
            # is this a mat * p
            else:
                return self.pointMatrixProduct(items)
        
        
        
    def _plusMinusAverage(self, op, items):
        """
        PEMDAS plus, minus, average operations
        """
    
        mat0 = self._isMatrixAttr(items[0])
        mat1 = self._isMatrixAttr(items[1])
    
        if not mat0 and not mat1:

            # plus
            if op == '+':
                node = self._createNode('plusMinusAverage', ss=True, name='addition1')
                mc.setAttr('%s.operation' % node, 1)
    
            # minus
            elif op == '-':
                node = self._createNode('plusMinusAverage', ss=True, name='subtraction1')
                mc.setAttr('%s.operation' % node, 2)
    
            # average
            elif op == 'avg':
                node = self._createNode('plusMinusAverage', ss=True, name='average1')
                mc.setAttr('%s.operation' % node, 3)
    
            else:
                raise Exception('unsupported operator: %s' % op)

    
            # Force single output if both inputs are single numerics
            counts = self._getPlugs(items, compound=False)
            if all(len(x) == 1 for x in counts):
                for i, obj in enumerate(items):
                    self._connectAttr(obj, '%s.input1D[%s]' % (node, i))
    
                return '%s.output1D' % node
    
            # Connect
            for i, obj in enumerate(items):
                self._connectAttr(obj, '%s.input3D[%s]' % (node, i))
    
            return '%s.output3D' % node
    
    
        else:
            
            if op == '+':
                return self.matrixAdd(items)
            
            elif op == 'avg':
                return self.matrixWeightedAdd(items)
                
            else:
                raise Exception('Unsupported %s operator used on matrices.'%op)
    
    
    @parsedcommand
    def power(self, items):
        """
        Return x raised to the power y
        """
        return self._multiplyDivide('**', items)
                
    @parsedcommand            
    def mult(self, items):
        """
        Multiplies two or more items
        """
        return self._multiplyDivide('*', items)
                
    @parsedcommand            
    def div(self, items):
        """
        Divides two or more items
        """        
        return self._multiplyDivide('/', items)
    
    @parsedcommand
    def add(self, items):
        """
        Adds two or more items
        """
        return self._plusMinusAverage('+', items)   
        
    @parsedcommand    
    def sub(self, items):
        """
        Subtracts two or more items
        """
        return self._plusMinusAverage('-', items)   
            
    @parsedcommand        
    def avg(self, items):
        """
        Averages two or more items
        """
        return self._plusMinusAverage('avg', items)
    


    
    # -------------------------- COMMON MATH FUNCTIONS --------------------------- #
    
    @parsedcommand
    def inv(self, items):
        """ 
        inv(<input>)
        
            Creates a network to yields an inverse (0.0-x) mirror operation.
        
            Examples
            --------
            >>> inv(pCube1.t)
        """
        if len(items) != 1:
            raise Exception('inverse() requires 1 input, given: %s' % items)
    
        if self._isMatrixAttr(items[0]):
            node = self._createNode('inverseMatrix', ss=True)
            self._connectAttr(items[0], '%s.inputMatrix' % node)
            return '%s.outputMatrix' % node
        else:
            return self.sub([self._long(0), items[0]])    
    
    
    @parsedcommand
    def rev(self, items):
        """ 
        rev(<input>)
        
            Creates a reverse node to do a (1.0-x) operation.
        
            Examples
            --------
            >>> rev(pCube1.t)
        """
        if len(items) != 1:
            raise Exception('rev() requires 1 input, given: %s' % items)
    
        node = self._createNode('reverse', ss=True)
        self._connectAttr(items[0], '%s.input' % node)
    
        return '%s.output' % node    
    
        
    @parsedcommand
    def sum(self, items):
        """ 
        sum(<input>, <input>, <input>, ...)
        
            Single node operation to sum all items in the list.
        
            Examples
            --------
            >>> sum(pCube1.t, pCube2.t, pCube3.t, pCube4.t)
        """
    
        if len(items) < 2:
            raise Exception('sum() requires minimum 2 inputs, given: %s' % items)
    
        return self.add(items)        
        
    @parsedcommand    
    def int(self, items):
        """ 
        int(<input>)
        
            Turns a float value(s) into an int.
        
            Examples
            --------
            >>> int(pCube1.t)
            >>> int(pCube1.tx)
        """
        
        exp = Expression(container='int1', debug=self.debug)
            
        if len(items) != 1:
            raise Exception('int() requires 1 input, given: %s' % items)
    
        node = None
        if len(self._getPlugs(items[0])[0]) > 1:
            node = exp._long3()
        else:
            node = exp._long()
    
        obj = items[0]
        exp('$f = 0.4999999')
        exp('$true  = $obj - $f', variables=locals())
        exp('$false = $obj + $f', variables=locals())
        exp('$node  = if ($obj > 0, $true, $false)', variables=locals())
        
        self.nodes.extend(exp.getNodes())    
        
        return node
    
        
    
    
    @parsedcommand
    def max(self, items):
        """ 
        max(<input>, <input>, <input>, ...)
        
            Returns the highest value in the list of inputs.
        
            Examples
            --------
            >>> max(pCube1.t, pCube2.t, pCube3.t, pCube4.t)
        """
    
        if len(items) < 2:
            raise Exception('max() requires minimum 2 inputs, given: %s' % items)
    
        ret = items[0]
        for obj in items[1:]:
            ret = self.cond([ret, '>', obj, ret, obj])
    
        return ret
    
    @parsedcommand
    def min(self, items):
        """ 
        min(<input>, <input>, <input>, ...)
        
            Returns the lowest value in the list of inputs.
        
            Examples
            --------
            >>> min(pCube1.t, pCube2.t, pCube3.t, pCube4.t)
        """
    
        if len(items) < 2:
            raise Exception('min() requires minimum 2 inputs, given: %s' % items)
    
        ret = items[0]
        for obj in items[1:]:
            ret = self.cond([ret, '<', obj, ret, obj])
    
        return ret
    
    @parsedcommand
    def exp(self, items):
        """ 
        exp(<input>)
        
            Return e raised to the power x, 
            where e is the base of natural logarithms (2.718281...)
        
            Examples
            --------
            >>> exp(pCube1.tx)
        """
        return self.power([self._double(math.e), items[0]])
    
    
    @parsedcommand
    def sign(self, items):
        """ 
        sign(<input>)
        
            Returns -1 for values < 0. +1 for values >= 0.
        
            Examples
            --------
            >>> sign(pCube1.t)
            >>> sign(pCube1.tx)
        """
        if len(items) != 1:
            raise Exception('sign() requires 1 input, given: %s' % items)
    
        return self.cond([items[0], '<', self._long(0), self._long(-1), self._long(1)])    
    
    
    @parsedcommand
    def floor(self, items):
        """ 
        floor(<input>)
        
            Returns the floor value of the input.
        
            Examples
            --------
            >>> floor(pCube1.t)
            >>> floor(pCube1.tx)
        """
        if len(items) != 1:
            raise Exception('floor() requires 1 input, given: %s' % items)
    
        # TODO _getPlug HACK :(, look into it for single input
        # THIS MIGHT BE CAUSEING ISSUES ELSEWHERE
        if len(self._getPlugs(items[0], compound=False)[0]) > 1:
            node = self._long3()
        else:
            node = self._long()
    
        f = self._double(0.4999999)  # correct Maya's inappropriate int convention
        floor = self.sub([items[0], f])
        self._connectAttr(floor, node)
    
        return node  
    
    @parsedcommand
    def ceil(self, items):
        """ 
        ceil(<input>)
        
            Returns the ceil value of the input.
        
            Examples
            --------
            >>> ceil(pCube1.t)
            >>> ceil(pCube1.tx)
        """
    
        if len(items) != 1:
            raise Exception('floor() requires 1 input, given: %s' % items)
    
        if len(self._getPlugs(items)) > 1:
            node = self._long3()
        else:
            node = self._long()
    
        f = self._double(0.4999999)  # corrent Maya's inappropriate int convention
        floor = self.add([items[0], f])
        self._connectAttr(floor, node)
    
        return node       
        
    
    @parsedcommand    
    def dist(self, items):
        """ 
        dist(<input>, <input>)
        
            Creates a distanceBetween node to find distance between points or matrices.
        
            Examples
            --------
            >>> dist(pCube1.t, pCube2.t)
            >>> dist(pCube1.wm, pCube2.wm)
        """

        if len(items) != 2:
            raise Exception('clamp requires 2 inputs, given: %s' % items)
    
        node = self._createNode('distanceBetween', ss=True)
    
        if self._isMatrixAttr(items[0]):
            self._connectAttr(items[0], '%s.inMatrix1' % node)
        else:
            self._connectAttr(items[0], '%s.point1' % node)
    
        if self._isMatrixAttr(items[1]):
            self._connectAttr(items[1], '%s.inMatrix2' % node)
        else:
            self._connectAttr(items[1], '%s.point2' % node)
    
        return '%s.distance' % node    
        

    @parsedcommand
    def abs(self, items):
        """ 
        abs(<input>)
        
            Outputs the absolute value of a float or vector.
        
            Examples
            --------
            >>> abs(pCube1.t)
            >>> abs(pCube1.tx)
        """
    
        items = self._getPlugs(items, compound=False)[0]
    
        if not len(items) in [1, 3]:
            raise Exception('abs works on 1 or 3 inputs, given: %s' % items)
    
    
        exp = Expression(container='abs1', debug=self.debug)
        exp('$zero = 0')
        exp('$neg1 = -1')

        result = []
        for item in items:
            exp('$neg = $neg1 * $item', variables=locals())
            test = exp('if ($item < $zero, $neg, $item)\n', variables=locals())[0]
            result.append(test)
    
        if len(result) > 1:
            result = exp('vector($result)\n', variables=locals())
    

        self.nodes.extend(exp.getNodes())  
        return result[0]
        
        
        
    @parsedcommand   
    def choice(self, items):
        """ 
        choice(<selector>, <input>, <input>, ...)
        
            Creates a choice node out of inputs.
            If selector is None, nothing will be set.
        
            Examples
            --------
            >>> choice(pCube1.someEnum, pCube2.wm, pCube3.wm)
            >>> choice(None, pCube2.wm, pCube3.wm) # leaves selector unplugged.
        """

        if len(items) < 2:
            raise Exception('choice requires minimum 2 inputs, given: %s' % items)
    
        # create choice node
        node = self._createNode('choice', ss=True)
    
        # plug selector
        if not items[0] in [None, 'None']:
            
            if self._isLongAttr(items[0]):
                self._connectAttr(items[0], '%s.selector' % node)
                
            else:
                integer = self('int(%s)'%items[0])[0]
                self._connectAttr(integer, '%s.selector' % node)
    
        # plug inputs
        for item in items[1:]:
            self._connectAttr(item, '%s.input' % node)
    
        return '%s.output' % node
    
    
    @parsedcommand
    def vector(self, items):
        """ 
        vector(<input>, <input>, <input>)
        
            Constructs a vector out of inputs.
        
            Examples
            --------
            >>> vector(pCube1.tx, pCube2.ty, pCube3.tz)
        """
        node = self._double3()
        for i, xyz in enumerate(['%s%s' % (node, x) for x in 'XYZ']):
    
            # skip 'None'
            if not items[i] in [None, 'None']:
                self._connectAttr(items[i], xyz)
    
        return node    
    
    
    
    
    # -------------------------- ALGEBRA MATH FUNCTIONS -------------------------- #
    
    # TODO: support Matrix dot product?
    @parsedcommand
    def _vectorProduct(self, name, op, normalize, items):
        """
        Vector Product wrapper
        """
        if len(items) != 2:
            raise Exception('%s requires 2 inputs, given: %s' % (name, items))
    
        node = self._createNode('vectorProduct', ss=True)
        mc.setAttr('%s.operation' % node, op)
        mc.setAttr('%s.normalizeOutput' % node, normalize)
    
        self._connectAttr(items[0], '%s.input1' % node)
        self._connectAttr(items[1], '%s.input2' % node)
    
        return '%s.output' % node            
    
    
    @parsedcommand
    def dot(self, items):
        """ 
        dot(<input>, <input>)
        
            Uses a vectorProduct to do a dot product between two vector inputs.
        
            Examples
            --------
            >>> dot(pCube1.t, pCube2.t)
        """
        return self._vectorProduct('dot', 1, 0, items)
        
        
    @parsedcommand
    def dotNormalized(self, items):
        """ 
        dotNormalized(<input>, <input>)
        
            Uses a normalized vectorProduct to do a dot product between two vector inputs.
        
            Examples
            --------
            >>> dotNormalized(pCube1.t, pCube2.t)
        """
        return self._vectorProduct('dotNormalized', 1, 1, items)
    
    
    @parsedcommand
    def cross(self, items):
        """ 
        cross(<input>, <input>)
        
            Uses a vectorProduct to do a cross product between two vector inputs.
        
            Examples
            --------
            >>> cross(pCube1.t, pCube2.t)
        """
        return self._vectorProduct('cross', 2, 0, items)
    
    
    @parsedcommand
    def crossNormalized(self, items):
        """ 
        crossNormalized(<input>, <input>)
        
            Uses a normalized vectorProduct to do a cross product between two vector inputs.
        
            Examples
            --------
            >>> crossNormalized(pCube1.t, pCube2.t)
        """
    
        return self._vectorProduct('crossNormalized', 2, 1, items)
    
    
    @parsedcommand
    def unit(self, items):
        """ 
        unit(<input>)
        
            Creates a network that yields a unit vector.
        
            Examples
            --------
            >>> unit(pCube1.t)
        """
    
        if len(items) != 1:
            raise Exception('unit() requires 1 input, given: %s' % items)
    
        exp    = Expression(container='unit1', debug=self.debug)
        mag    = exp.mag(items)
        mult   = exp.div([items[0], mag])
        zero   = exp._long(0)
        two    = exp._long(2)
    
        node   = mult.split('.')[0]
        test   = exp.cond([mag, '==', zero, zero, two])
        bypass = exp.cond([mag, '==', zero, test, mult])
    
        self.nodes.extend(exp.getNodes())    
    
        self._connectAttr(test, '%s.operation' % node) # silence div by zero error
        return bypass      
    
    
    
    # ------------------------- TRIGONOMETRIC FUNCTIONS -------------------------- # 

    def _trigonometry(self, items, x, y, modulo=None, container=None):
        """
        Sets up a ramapValue node for sine/cosine trigonometric functions.
        """
        items = self._getPlugs(items, compound=False)
        
        exp = Expression(container=container, debug=self.debug)
        results = []
        for i in range(len(items[0])):
    
            plug = items[0][i]
            if modulo:
                plug = exp(str(plug) + '%' + str(modulo))[0]
    
            node = exp._createNode('remapValue', ss=self.debug)
            self._connectAttr(plug, '%s.inputValue' % node)
    
            for j in range(len(x)):
                mc.setAttr('%s.value[%s].value_Position' % (node, j), x[j])
                mc.setAttr('%s.value[%s].value_FloatValue' % (node, j), y[j])
                mc.setAttr('%s.value[%s].value_Interp' % (node, j), 2)
    
            results.append('%s.outValue' % node)
    
    
        result = None
        if len(results) == 1:
            result = results[0]
    
        elif len(results) == 3:
            vec = exp._double3()
            
            for i, xyz in enumerate(['X', 'Y', 'Z']):
                mc.connectAttr(results[i], '%s%s' % (vec, xyz), f=True)
                
            result = vec
    
        else:
            raise Exception('trigonometric functions ony supports 1 or 3 plugs')
        
        

        self.nodes.extend(exp.getNodes())           
        
        return result
        
        
    @parsedcommand    
    def degrees(self, items):
        """
        degrees(<input>)
        
            Converts incomming values from radians to degrees.
            (obj in radians * 57.29577951)
        
            Examples
            --------
            >>> degrees(radians(pCube1.rx)) # returns a network which converts rotationX to radians and back to degrees.
            >>> degrees(radians(pCube1.r))  # returns a network which converts [rx, ry, rz] to radians and back to degrees.
        """
        #return self.eval('%s * %s' % (items[0], (180./math.pi) ))
        exp = Expression(container='degrees', debug=self.debug)
        result = exp('%s * %s' % (items[0], (180./math.pi) ))[0]
        
        self.nodes.extend(exp.getNodes())
        return result


    @parsedcommand
    def radians(self, items):
        """ 
        radians(<input>)
        
            Converts incomming values from degrees to radians.
            (input in degrees * 0.017453292)
        
            Examples
            --------
            >>> radians(pCube1.rx) # returns a network which converts rotationX to radians.
            >>> radians(pCube1.r)  # returns a network which converts [rx, ry, rz] to radians.
        """
        #return self.eval('%s * %s' % (items[0], (math.pi/180.) ))
        exp = Expression(container='radians', debug=self.debug)
        result = exp('%s * %s' % (items[0], (math.pi/180.) ))[0]
        
        self.nodes.extend(exp.getNodes()) 
        return result    


    # TODO: add built in start, stop remap values
    @parsedcommand
    def easeIn(self, items):
        """ 
        easeIn(<input>)
        
            Creates an easeIn "tween" function.
        
            Examples
            --------
            >>> easeIn(pCube1.tx) # returns a network which tweens pCube1's translateX value.
            >>> easeIn(pCube1.t)  # returns a network which tweens pCube1's [tx, ty, tz] values.
        """
        return self._trigonometry(items, x=[0, 1], y=[0, 1], container='easeIn')


    # TODO: add built in start, stop remap values
    @parsedcommand
    def easeOut(self, items):
        """ 
        easeOut(<input>)
        
            Creates an easeIn "tween" function.
        
            Examples
            --------
            >>> easeOut(pCube1.tx) # returns a network which tweens pCube1's translateX value.
            >>> easeOut(pCube1.t)  # returns a network which tweens pCube1's [tx, ty, tz] values.
        """
        return self._trigonometry(items, x=[1, 0], y=[0, 1], container='easeOut')

    @parsedcommand
    def sin(self, items):
        """ 
        sin(<input>)
        
            Creates a sine function (in radians).
        
            Examples
            --------
            >>> sin(pCube1.tx) # returns a network which passes pCube1's translateX into a sine function.
            >>> sin(pCube1.t)  # returns a network which passes pCube1's [tx, ty, tz] into a sine functions.
        """
        x = [(-5 * math.pi / 2), (5 * math.pi / 2), (-3 * math.pi / 2), (-1 * math.pi / 2), (math.pi / 2), (3 * math.pi / 2)]
        y = [-1, 1, 1, -1, 1, -1]
        return self._trigonometry(items, x, y, modulo=2 * math.pi, container='sin1')

    @parsedcommand
    def sind(self, items):
        """ 
        sind(<input>)
        
            Creates a sine function (in degrees).
        
            Examples
            --------
            >>> sind(pCube1.tx) # returns a network which passes pCube1's translateX into a sine function.
            >>> sind(pCube1.t)  # returns a network which passes pCube1's [tx, ty, tz] into a sine functions.
        """
        x = [(-5 * 180. / 2), (5 * 180. / 2), (-3 * 180. / 2), (-1 * 180. / 2), (180. / 2), (3 * 180. / 2)]
        y = [-1, 1, 1, -1, 1, -1]        
        return self._trigonometry(items, x, y, modulo=2 * 180., container='sind1')

    @parsedcommand
    def cos(self, items):
        """ 
        cos(<input>)
        
            Creates a cosine function (in radians).
        
            Examples
            --------
            >>> cos(pCube1.tx) # returns a network which passes pCube1's translateX into a cosine function.
            >>> cos(pCube1.t)  # returns a network which passes pCube1's [tx, ty, tz] into a cosine functions.
        """
    
        x = [(-2 * math.pi), (2 * math.pi), (-1 * math.pi), 0, math.pi]
        y = [1, 1, -1, 1, -1]
        return self._trigonometry(items, x, y, modulo=2 * math.pi, container='cos1')    
    
    @parsedcommand
    def cosd(self, items):
        """ 
        cosd(<input>)
        
            Creates a cosine function (in degrees).
        
            Examples
            --------
            >>> cosd(pCube1.tx) # returns a network which passes pCube1's translateX into a cosine function.
            >>> cosd(pCube1.t)  # returns a network which passes pCube1's [tx, ty, tz] into a cosine functions.
        """
        x = [(-2 * 180.), (2 * 180.), (-1 * 180.), 0, 180.]
        y = [1, 1, -1, 1, -1]
        return self._trigonometry(items, x, y, modulo=2 * 180., container='cosd1')   
    
    
    @parsedcommand
    def acos(self, items):
        """ 
        acos(<input>)
        
            Approximates an arc cosine function (in radians).
        
            Examples
            --------
            >>> acos(pCube1.tx) # returns a network which passes pCube1's translateX into an arc cosine approximation function.
            >>> acos(pCube1.t)  # returns a network which passes pCube1's [tx, ty, tz] into an arc cosine approximation functions.
        """
    
        # https://developer.download.nvidia.com/cg/acos.html
        items = self._getPlugs(items, compound=False)
        results = []
        e = Expression(container='acos1', debug=self.debug)
         
        for i in range(len(items[0])):
            plug = items[0][i]
    
            exp = """
            $negate = if(%s<0,1,0)
            $x   = abs(%s)
            $ret = -0.0187293
            $ret = $ret + 0.0742610
            $ret = $ret * $x
            $ret = $ret - 0.2121144
            $ret = $ret * $x
            $ret = $ret + 1.5707288
            $ret = $ret * (1.0-$x)**0.5
            $ret = $ret - 2 * $negate * $ret
            $negate * 3.14159265358979 + $ret
            """ % (plug, plug)
    
            # pack in it's own container to reduce the clutter
            results.append(e(exp)[0])
            #results.append(self(exp))
    
        result = None
        if len(results) == 1:
            result = results[0]
    
        elif len(results) == 3:
            result = e._double3()
            for i, xyz in enumerate(['X', 'Y', 'Z']):
                self._connectAttr(results[i], '%s%s' % (result, xyz))
    
        else:
            raise Exception('trigonometric functions only supports 1 or 3 plugs')    

        self.nodes.extend(e.getNodes())
        return result



    @parsedcommand
    def asin(self, items):
        """ 
        asin(<input>)
        
            Approximates an arc sine function (in radians).
        
            Examples
            --------
            >>> asin(pCube1.tx) # returns a network which passes pCube1's translateX into an arc sine approximation function.
            >>> asin(pCube1.t)  # returns a network which passes pCube1's [tx, ty, tz] into an arc sine approximation functions.
        """
    
        # https://developer.download.nvidia.com/cg/asin.html
        items = self._getPlugs(items, compound=False)
        e = Expression(container='asin1', debug=self.debug)
        results = []
        for i in range(len(items[0])):
            plug = items[0][i]
    
            exp = """
            $negate = if(%s<0,1,0)
            $x   = abs(%s)
            $ret = -0.0187293
            $ret = $ret * $x
            $ret = $ret + 0.0742610
            $ret = $ret * $x
            $ret = $ret - 0.2121144
            $ret = $ret * $x
            $ret = $ret + 1.5707288
            $ret = 3.14159265358979*0.5 - (1-$x)**0.5 * $ret
            
            $ret - 2 * $negate * $ret
            """ % (plug, plug)
            
            # pack in it's own container to reduce the clutter
            e = Expression(container='asin1', debug=self.debug)
            results.append(e(exp)[0])
            #results.append(self(exp))
            

        result = None
        if len(results) == 1:
            result = results[0]
    
        elif len(results) == 3:
            result = e._double3()
            for i, xyz in enumerate(['X', 'Y', 'Z']):
                self._connectAttr(results[i], '%s%s' % (result, xyz))
    
        else:
            raise Exception('trigonometric functions only supports 1 or 3 plugs')    
    
        self.nodes.extend(e.getNodes())
        return result
        
        
        

    @parsedcommand
    def acosd(self, items):
        """ 
        acosd(<input>)
        
            Approximates an arc cosine function (in degrees).
        
            Examples
            --------
            >>> acosd(pCube1.tx) # returns a network which passes pCube1's translateX into an arc cosine approximation function.
            >>> acosd(pCube1.t)  # returns a network which passes pCube1's [tx, ty, tz] into an arc cosine approximation functions.
        """
        
    
        # pack in it's own container to reduce the clutter
        e = Expression(container='acosd1', debug=self.debug)
        result = e('degrees(acos(%s))' % items[0])[0]
        self.nodes.extend(e.getNodes())
        return result
        #return self.eval('degrees(acos(%s))' % items[0])
    
    
    
    @parsedcommand
    def asind(self, items):
        """ 
        asind(<input>)
        
            Approximates an arc sine function (in radians).
        
            Examples
            --------
            >>> asind(pCube1.tx) # returns a network which passes pCube1's translateX into an arc sine approximation function.
            >>> asind(pCube1.t)  # returns a network which passes pCube1's [tx, ty, tz] into an arc sine approximation functions.
        """
        e = Expression(container='asind1', debug=self.debug)
        result = e('degrees(asin(%s))' % items[0])[0]
        self.nodes.extend(e.getNodes())
        return result        
        #return self.eval('degrees(asin(%s))' % items[0])    
    
    
    @parsedcommand
    def tan(self, items):
        """ 
        tan(<input>)
        
            Approximates a tan function (in radians).
        
            Examples
            --------
            >>> tan(pCube1.tx) # returns a network which passes pCube1's translateX into a tan approximation function.
            >>> tan(pCube1.t)  # returns a network which passes pCube1's [tx, ty, tz] into a tan approximation functions.
        """
    
        # https://developer.download.nvidia.com/cg/tan.html
        exp = """
        $sin     = sin(%s)
        $cos     = cos(%s)
        $divtest = if($cos != 0, $cos, 1)
        $tan     = $sin/$divtest
        if($cos != 0, $tan, 16331239353195370)
        """ % (items[0], items[0])
        
        # pack in it's own container to reduce the clutter
        e = Expression(container='tan1', debug=self.debug)
        result = e(exp)[0]
        self.nodes.extend(e.getNodes())
        return result
        #results.append(self(exp))
        
    
    @parsedcommand
    def tand(self, items):
        """ 
        tand(<input>)
        
            Approximates a tan function (in degrees).
        
            Examples
            --------
            >>> tand(pCube1.tx) # returns a network which passes pCube1's translateX into a tan approximation function.
            >>> tand(pCube1.t)  # returns a network which passes pCube1's [tx, ty, tz] into a tan approximation functions.
        """
    
        exp = """
        $sin     = sind(%s)
        $cos     = cosd(%s)
        $divtest = if($cos != 0, $cos, 1)
        $tan     = $sin/$divtest
        if($cos != 0, $tan, 16331239353195370)
        """ % (items[0], items[0])
    
        # pack in it's own container to reduce the clutter
        e = Expression(container='tand1', debug=self.debug)
        result = e(exp)[0]
        self.nodes.extend(e.getNodes)
        return result    
        #return self.eval(exp)
        
    
    
    # TODO
    #def _atan2(items):
        #
        # float2 atan2(float2 y, float2 x)
        # {
        #  float2 t0, t1, t2, t3, t4;
        #
        #  t3 = abs(x);
        #  t1 = abs(y);
        #  t0 = max(t3, t1);
        #  t1 = min(t3, t1);
        #  t3 = float(1) / t0;
        #  t3 = t1 * t3;
        #
        #  t4 = t3 * t3;
        #  t0 =         - float(0.013480470);
        #  t0 = t0 * t4 + float(0.057477314);
        #  t0 = t0 * t4 - float(0.121239071);
        #  t0 = t0 * t4 + float(0.195635925);
        #  t0 = t0 * t4 - float(0.332994597);
        #  t0 = t0 * t4 + float(0.999995630);
        #  t3 = t0 * t3;
        #
        #  t3 = (abs(y) > abs(x)) ? float(1.570796327) - t3 : t3;
        #  t3 = (x < 0) ?  float(3.141592654) - t3 : t3;
        #  t3 = (y < 0) ? -t3 : t3;
        #
        #  return t3;
        # }    
    
    
    # TODO
    #def _atan(items):
        #
        # float atan(float x) {
        #    return _atan2(x, float(1));
        # }    





    # --------------------------- QUATERNION FUNCTIONS --------------------------- #

    def _quaternion(self, items, quat_node, sequential=False, output_attr='outputQuat'):
        """ 
        Quaternion processor utility used by most quaternion functions.
        """
        
        # make sure given items are lists, tuples or sets
        if not isinstance(items, (list, tuple, set)):
            items = [items]
            
        if not sequential:
            if len(items) != 1:
                raise Exception('%s requires 1 input, given: %s' % (quat_node, items))
        else:
            if len(items) < 2:
                raise Exception('%s requires multiple inputs, given: %s' % (quat_node, items))
    
        # Test inputs for quaternions, if matrix given
        # do a conversion for convenience.
        for i, item in enumerate(items):
            if self._isMatrixAttr(item):
                items[i] = self.matrixToQuat(item)
    
            elif not self._isQuatAttr(item):
                raise Exception('%s requires quaternions, given: %s' % (quat_node, items))
    
        node = self._createNode(quat_node, ss=True)
        if sequential:
            self._connectAttr(items[0], '%s.input1Quat' % node)
            self._connectAttr(items[1], '%s.input2Quat' % node)
    
            for item in items[2:]:
                node_ = self._createNode(quat_node, ss=True)
                self._connectAttr('%s.outputQuat' % node, '%s.input1Quat' % node_)
                self._connectAttr(item, '%s.input2Quat' % node_)
                node = node_
    
        else:
            self._connectAttr(items[0], '%s.inputQuat' % node)
    
        return '%s.%s' % (node, output_attr)
    
    @parsedcommand
    def quatAdd(self, items):
        """ 
        quatAdd(<input>, <input>, <input>, ...)
        
            Returns the sum of added quaternions.
        
            Examples
            --------
            >>> quatAdd(pCube1.rq, pCube1.rq)
        """
        return self._quaternion(items, 'quatAdd', sequential=True)
    
    @parsedcommand
    def quatProd(self, items):
        """ 
        quatProd(<input>, <input>, <input>, ...)
        
            Returns the product of multiplied quaternions.
        
            Examples
            --------
            >>> quatProd(pCube1.rq, pCube2.rq)
        """
        return self._quaternion(items, 'quatProd', sequential=True)
    
    @parsedcommand
    def quatSub(self, items):
        """ 
        quatSub(<input>, <input>, <input>, ...)
        
            Returns the sum of subtracted quaternions.
        
            Examples
            --------
            >>> quatSub(pCube1.rq, pCube1.rq)
        """
        return self._quaternion(items, 'quatSub', sequential=True)
    
    @parsedcommand
    def quatNegate(self, items):
        """ 
        quatNegate(<input>)
        
            Negates a quaternion.
        
            Examples
            --------
            >>> quatNegate(pCube1.wm)
        """
        return self._quaternion(items, 'quatNegate')
    
    @parsedcommand
    def quatToEuler(self, items):
        """ 
        quatToEuler(<input>)
        
            Turns a quaternion into a euler angle.
        
            Examples
            --------
            >>> quatToEuler(pCube1.wm)
        """
        return self._quaternion(items, 'quatToEuler', output_attr='outputRotate')
    
    @parsedcommand
    def eulerToQuat(self, items):
        """ 
        eulerToQuat(<euler>,<rotateOrder>)
        
            Turns a euler angle into a guaternion.
        
            Examples
            --------
            >>> eulerToQuat(pCube1.r, some_node.ro)
            >>> eulerToQuat(pCube1.r)
        """
        
        if len(items) > 2:
            raise Exception('mag requires max 2 inputs, given: %s' % items)
    
        node = self._createNode('eulerToQuat', ss=True)
        self._connectAttr(items[0], '%s.inputRotate' % node)
        
        if len(items) == 2:
            self._connectAttr(items[1], '%s.inputRotateOrder' % node)
        else:
            
            # autoconnect rotate order if present
            obj = items[0].split('.')[0]
            
            if mc.attributeQuery('rotateOrder', node=obj, exists=True):
                mc.connectAttr('%s.ro' % obj, '%s.inputRotateOrder' % node)        
    
        return '%s.outputQuat' % node
    
    @parsedcommand    
    def quatNormalize(self, items):
        """ 
        quatNormalize(<input>)
        
            Normalizes a quaternion.
        
            Examples
            --------
            >>> quatNormalize(pCube1.wm)
        """
        return self._quaternion(items, 'quatNormalize')
    
    @parsedcommand
    def quatInvert(self, items):
        """ 
        quatInvert(<input>)
        
            Inverts a quaternion.
        
            Examples
            --------
            >>> quatInvert(pCube1.wm)
        """
        return self._quaternion(items, 'quatInvert')
    
    @parsedcommand
    def quatConjugate(self, items):
        """ 
        quatConjugate(<input>)
        
            Conjugates a quaternion.
        
            Examples
            --------
            >>> quatConjugate(pCube1.wm)
        """
        return self._quaternion(items, 'quatConjugate')
    
    @parsedcommand
    def quatSlerp(self, items):
        """ 
        quatSlerp(<input>, <input>, ...)
        
            Slerps between two quaternions with optional weight values.
            (default = 0.5)
        
            Examples
            --------
            >>> quatSlerp(pCube1.wm, pCube2.wm)
            >>> quatSlerp(pCube1.wm, pCube2.wm, pCube1.weight)
            
        """
        if len(items) <= 1:
            raise Exception('quatSlerp requires 2 or more inputs, given: %s' % items)
    
        # parse inputs between matrices and weights
        quats = []
        weights = []
    
        for item in items:
    
            # is this a matrix?
            if self._isMatrixAttr(item):
                quats.append(self.matrixToQuat(item))
    
            elif self._isQuatAttr(item):
                quats.append(item)
    
            # assume this is a weight
            else:
                weights.append(item)
    
        node = self._createNode('quatSlerp', ss=True)
    
        self._connectAttr(quats[0], '%s.input1Quat' % (node))
        self._connectAttr(quats[1], '%s.input2Quat' % (node))
    
        # if no weights provided, set T to 0.5
        if not weights:
            # weights.append(self._double(0.5))
            # self._connectAttr(weights[0], '%s.inputT'% (node))
            mc.setAttr('%s.inputT' % (node), 0.5)
            
        else:
            self._connectAttr(weights[0], '%s.inputT'% (node))
    
        return '%s.outputQuat' % node




    # ----------------------------- MATRIX FUNCTIONS ----------------------------- #

    def _matrix(self, items, matrix_node, output_attr='outputMatrix'):
        """ 
        Matrix processor utility.
        """
    
        # make sure given items are lists, tuples or sets
        if not isinstance(items, (list, tuple, set)):
            items = [items]
    
        # test input count
        if len(items) != 1:
            raise Exception('%s requires 1 input, given: %s' % (matrix_node, items))
    
        # make sure items are matrices
        for item in items:
            if not self._isMatrixAttr(item):
                raise Exception('%s requires matrices, given: %s' % (matrix_node, items))
    
        # process item
        node = self._createNode(matrix_node, ss=True)
        self._connectAttr(items[0], '%s.inputMatrix' % node)
    
        return '%s.%s' % (node, output_attr)
    
    @parsedcommand
    def matrixInverse(self, items):
        """ 
        matrixInverse(<input>)
        
            Returns the inverse matrix.
        
            Examples
            --------
            >>> matrixInverse(pCube1.wm)
        """
        return self._matrix(items, 'inverseMatrix')
    
    @parsedcommand
    def matrixTranspose(self, items):
        """ 
        matrixTranspose(<input>)
        
            Returns the transposed matrix.
        
            Examples
            --------
            >>> matrixTranspose(pCube1.wm)
        """
        return self._matrix(items, 'transposeMatrix')
    
    @parsedcommand
    def matrixToQuat(self, items):
        """ 
        matrixToQuat(<input>)
        
            Converts a matrix into a quaternion.
        
            Examples
            --------
            >>> matrixToQuat(pCube1.wm)
        """
        return self._matrix(items, 'decomposeMatrix', output_attr='outputQuat')
    
    @parsedcommand
    def matrix(self, items):
        """ 
        matrix(<input>, <input>, <input>, <input>)
        
            Constructs a matrix from a list of up to 4 vectors (X,Y,Z,position)
        
            Examples
            --------
            >>> matrix(pCube1.t, pCube2.t, pCube3.t)
            >>> matrix(pCube1.t, pCube2.t, pCube3.t, pCube4.t)
        """
    
        if len(items) > 4:
            raise Exception('matrix constructor accepts up to 4 inputs, given: %s' % items)
    
        items = self._getPlugs(items, compound=False)
    
        M = self._createNode('fourByFourMatrix', ss=True)
        for i in range(len(items)):
            for j in range(len(items[i])):
                if not items[i][j]  in [None, 'None']:
                    plug = '%s.in%s%s' % (M, i, j)
                    self._connectAttr(items[i][j], plug)
    
        return '%s.output' % M
    
    @parsedcommand
    def matrixCompose(self, items):
        """ 
        matrixCompose(<translate>, <rotate/quaternion>, <scale>, <shear> <rotateOrder>)
        
            Constructs a matrix from a list of up to 5 inputs.
        
            Examples
            --------
            >>> matrixCompose(pCube1.t, pCube1.r, pCube1.s, None) # pCube1's rotate order will be plugged in
            >>> matrixCompose(pCube1.t, eulerToQuat(pCube1.r), pCube1.s, None) # inputQuaternioon will be used
            >>> matrixCompose(pCube3.t) # identity matrix with just a position
        """
    
        if len(items) > 5:
            raise Exception('matrix composer accepts up to 5 inputs, given: %s' % items)
    
    
        node = self._createNode('composeMatrix', ss=True)
        plugs0 = ['inputTranslate', 'inputRotate', 'inputScale', 'inputShear', 'inputRotateOrder']
        plugs1 = ['inputTranslate', 'inputQuat',   'inputScale', 'inputShear', 'inputRotateOrder']
        
        for i, item in enumerate(items):
            if not item in [None, 'None']:
                plugs = self._listPlugs(item)
                
                # quaternion test
                if i == 1:
                    if len(plugs) == 5:
                        mc.setAttr('%s.useEulerRotation'%node, 0)
                        self._connectAttr(plugs[0], '%s.%s'%(node,plugs1[i]))
                        
                    else:
                        # autoconnect rotate order if present
                        self._connectAttr(item, '%s.%s'%(node, plugs0[i]))
                        obj = items[0].split('.')[0]
                        
                        if mc.attributeQuery('rotateOrder', node=obj, exists=True):
                            mc.connectAttr('%s.ro' % obj, '%s.inputRotateOrder' % node)                                
                    
                else:
                    self._connectAttr(item, '%s.%s'%(node, plugs0[i]))

        return '%s.outputMatrix' % node    
    
    @parsedcommand
    def matrixMult(self, items):
        """ 
        matrixMult(<input>, <input>, ...)
        
            Multiplies 2 or more matrices together.
        
            Examples
            --------
            >>> pCube1.wm * pCube2.wm
            >>> matrixMult(pCube1.wm, pCube2.wm, pCube3.wm)
        """
        if len(items) <= 1:
            raise Exception('matrixMult requires 2 or more inputs, given: %s' % items)
    
        for item in items:
            if not self._isMatrixAttr(item):
                raise Exception('matrixMult requires matrices, given: %s' % items)
    
        node = self._createNode('multMatrix', ss=True)
    
        for item in items:
            self._connectAttr(item, '%s.matrixIn' % node)
    
        return '%s.matrixSum' % node
    
    @parsedcommand
    def matrixAdd(self, items):
        """ 
        matrixAdd(<input>, <input>, ...)
        
            Adds matrices together.
        
            Examples
            --------
            >>> pCube1.wm + pCube2.wm
            >>> matrixAdd(pCube1.wm, pCube2.wm, pCube3.wm, ...)
        """
        if len(items) <= 1:
            raise Exception('matrixAdd requires 2 or more inputs, given: %s' % items)
    
        for item in items:
            if not self._isMatrixAttr(item):
                raise Exception('matrixAdd requires matrices, given: %s' % items)
    
        node = self._createNode('addMatrix', ss=True)
    
        for item in items:
            self._connectAttr(item, '%s.matrixIn' % node)
    
        return '%s.matrixSum' % node
    
    @parsedcommand
    def matrixWeightedAdd(self, items):
        """ 
        matrixWeightedAdd(<input>, <input>, ...)
        
            Adds matrices together with optional weight values.
            (default = averaged)
        
            Examples
            --------
            >>> matrixWeightedAdd(pCube1.wm, pCube2.wm, pCube3.wm, ...)
            >>> matrixWeightedAdd(pCube1.wm, pCube2.wm, pCube1.weight, pCube2.weight)
            
        """
        if len(items) <= 1:
            raise Exception('matrixAdd requires 2 or more inputs, given: %s' % items)
    
        # parse inputs between matrices and weights
        matrices = []
        weights = []
    
        for item in items:
    
            # is this a matrix?
            if self._isMatrixAttr(item):
                matrices.append(item)
    
            # assume this is a weight
            else:
                weights.append(item)
    
        # test weight inputs
        # if 0, then weights = 1/matrix count
        # if 1 and we have two matrices, then weight is 0-1 from mat1 to mat2
        # if n weight for n matrices, this is the normal use case
        # error out otherwise
        weight_count = len(weights)
        matrix_count = len(matrices)
    
        if weight_count == 0:
            w = 1. / matrix_count
            for _ in matrices:
                weights.append(self._double(w))
    
        elif matrix_count == 2 and weight_count == 1:
            weights.append(weights[0])
            weights[0] = self.rev([weights[-1]])
    
        elif matrix_count > 1 and weight_count != matrix_count:
            raise Exception('matrixWeightedAdd invalid inputs, given: %s' % items)
    
        node = self._createNode('wtAddMatrix', ss=True)
    
        for i in range(matrix_count):
            self._connectAttr(matrices[i], '%s.wtMatrix.matrixIn' % (node))
            self._connectAttr(weights[i], '%s.wtMatrix.weightIn' % (node))
    
        return '%s.matrixSum' % node
    
    
    @parsedcommand
    def vectorMatrixProduct(self, items):
        """ 
        vectorMatrixProduct(<input>, <input>)
        
            Creates a vectorProduct node to do a vector matrix product.
        
            Examples
            --------
            >>> pCube1.t * pCube2.wm
            >>> vectorMatrixProduct(pCube1.t, pCube2.wm)
        """
    
        if len(items) != 2:
            raise Exception('vectorMatrixProduct requires 2 inputs, given: %s' % items)
    
        node = self._createNode('vectorProduct', ss=True)
        mc.setAttr('%s.operation' % node, 3)
        mc.setAttr('%s.normalizeOutput' % node, 0)
    
        matrix0 = self._isMatrixAttr(items[0])
        matrix1 = self._isMatrixAttr(items[1])
    
        if matrix0 == matrix1:
            raise Exception('vectorMatrixProduct requires a matrix and a vector, given: %s' % items)
    
        if matrix0:
            self._connectAttr(items[0], '%s.matrix' % node)
        else:
            self._connectAttr(items[0], '%s.input1' % node)
    
        if matrix1:
            self._connectAttr(items[1], '%s.matrix' % node)
        else:
            self._connectAttr(items[1], '%s.input1' % node)
    
        return '%s.output' % node
    
    @parsedcommand
    def vectorMatrixProductNormalized(self, items):
        """ 
        vectorMatrixProductNormalized(<input>, <input>)
        
            Creates a normalized vectorProduct node to do a vector matrix product.
        
            Examples
            --------
            >>> vectorMatrixProductNormalized(pCube1.t, pCube2.wm)
        """
    
        if len(items) != 2:
            raise Exception('vectorMatrixProductNormalized requires 2 inputs, given: %s' % items)
    
        node = self._createNode('vectorProduct', ss=True)
        mc.setAttr('%s.operation' % node, 3)
        mc.setAttr('%s.normalizeOutput' % node, 1)
    
        matrix0 = self._isMatrixAttr(items[0])
        matrix1 = self._isMatrixAttr(items[1])
    
        if matrix0 == matrix1:
            raise Exception('nVectorMatrixProduct requires a matrix and a vector, given: %s' % items)
    
        if matrix0:
            self._connectAttr(items[0], '%s.matrix' % node)
        else:
            self._connectAttr(items[0], '%s.input1' % node)
    
        if matrix1:
            self._connectAttr(items[1], '%s.matrix' % node)
        else:
            self._connectAttr(items[1], '%s.input1' % node)
    
        return '%s.output' % node
    
    @parsedcommand
    def pointMatrixProduct(self, items):
        """ 
        pointMatrixProduct(<input>, <input>)
        
            Creates a vectorProduct node to do a point matrix product.
        
            Examples
            --------
            >>> pointMatrixProduct(pCube1.t, pCube2.wm)
        """
    
        if len(items) != 2:
            raise Exception('pointMatrixProduct requires 2 inputs, given: %s' % items)
    
        node = self._createNode('vectorProduct', ss=True)
        mc.setAttr('%s.operation' % node, 4)
        mc.setAttr('%s.normalizeOutput' % node, 0)
    
        matrix0 = self._isMatrixAttr(items[0])
        matrix1 = self._isMatrixAttr(items[1])
    
        if matrix0 == matrix1:
            raise Exception('pointMatrixProduct requires a matrix and a vector, given: %s' % items)
    
        if matrix0:
            self._connectAttr(items[0], '%s.matrix' % node)
        else:
            self._connectAttr(items[0], '%s.input1' % node)
    
        if matrix1:
            self._connectAttr(items[1], '%s.matrix' % node)
        else:
            self._connectAttr(items[1], '%s.input1' % node)
    
        return '%s.output' % node
    
    @parsedcommand
    def matrixAdd(self, items):
        """ 
        matrixAdd(<input>, <input>, ...)
        
            Adds matrices together.
        
            Examples
            --------
            >>> pCube1.wm + pCube2.wm
            >>> matrixAdd(pCube1.wm, pCube2.wm, pCube3.wm, ...)
        """
        if len(items) <= 1:
            raise Exception('matrixAdd requires 2 or more inputs, given: %s' % items)
    
        for item in items:
            if not self._isMatrixAttr(item):
                raise Exception('matrixAdd requires matrices, given: %s' % items)
    
        node = self._createNode('addMatrix', ss=True)
    
        for item in items:
            self._connectAttr(item, '%s.matrixIn' % node)
    
        return '%s.matrixSum' % node
    
    @parsedcommand
    def matrixWeightedAdd(self, items):
        """ 
        matrixWeightedAdd(<input>, <input>, ...)
        
            Adds matrices together with optional weight values.
            (default = averaged)
        
            Examples
            --------
            >>> matrixWeightedAdd(pCube1.wm, pCube2.wm, pCube3.wm, ...)
            >>> matrixWeightedAdd(pCube1.wm, pCube2.wm, pCube1.weight, pCube2.weight)
            
        """
        if len(items) <= 1:
            raise Exception('matrixAdd requires 2 or more inputs, given: %s' % items)
    
        # parse inputs between matrices and weights
        matrices = []
        weights = []
    
        for item in items:
    
            # is this a matrix?
            if self._isMatrixAttr(item):
                matrices.append(item)
    
            # assume this is a weight
            else:
                weights.append(item)
    
        # test weight inputs
        # if 0, then weights = 1/matrix count
        # if 1 and we have two matrices, then weight is 0-1 from mat1 to mat2
        # if n weight for n matrices, this is the normal use case
        # error out otherwise
        weight_count = len(weights)
        matrix_count = len(matrices)
    
        if weight_count == 0:
            w = 1. / matrix_count
            for _ in matrices:
                weights.append(self._double(w))
    
        elif matrix_count == 2 and weight_count == 1:
            weights.append(weights[0])
            weights[0] = self.rev([weights[-1]])
    
        elif matrix_count > 1 and weight_count != matrix_count:
            raise Exception('matrixWeightedAdd invalid inputs, given: %s' % items)
    
        node = self._createNode('wtAddMatrix', ss=True)
    
        for i in range(matrix_count):
            self._connectAttr(matrices[i], '%s.wtMatrix.matrixIn' % (node))
            self._connectAttr(weights[i], '%s.wtMatrix.weightIn' % (node))
    
        return '%s.matrixSum' % node
    
    @parsedcommand
    def matrixMult(self, items):
        """ 
        matrixMult(<input>, <input>, ...)
        
            Multiplies 2 or more matrices together.
        
            Examples
            --------
            >>> pCube1.wm * pCube2.wm
            >>> matrixMult(pCube1.wm, pCube2.wm, pCube3.wm)
        """
        if len(items) <= 1:
            raise Exception('matrixMult requires 2 or more inputs, given: %s' % items)
    
        for item in items:
            if not self._isMatrixAttr(item):
                raise Exception('matrixMult requires matrices, given: %s' % items)
    
        node = self._createNode('multMatrix', ss=True)
    
        for item in items:
            self._connectAttr(item, '%s.matrixIn' % node)
    
        return '%s.matrixSum' % node
    
    @parsedcommand
    def pointMatrixProductNormalized(self, items):
        """ 
        pointMatrixProductNormalized(<input>, <input>)
        
            Creates a normalized vectorProduct node to do a point matrix product.
        
            Examples
            --------
            >>> pointMatrixProductNormalized(pCube1.t, pCube2.wm)
        """
    
        if len(items) != 2:
            raise Exception('pointMatrixProductNormalized requires 2 inputs, given: %s' % items)
    
        node = self._createNode('vectorProduct', ss=True)
        mc.setAttr('%s.operation' % node, 4)
        mc.setAttr('%s.normalizeOutput' % node, 1)
    
        matrix0 = self._isMatrixAttr(items[0])
        matrix1 = self._isMatrixAttr(items[1])
    
        if matrix0 == matrix1:
            raise Exception('pointMatrixProductNormalized requires a matrix and a vector, given: %s' % items)
    
        if matrix0:
            self._connectAttr(items[0], '%s.matrix' % node)
        else:
            self._connectAttr(items[0], '%s.input1' % node)
    
        if matrix1:
            self._connectAttr(items[1], '%s.matrix' % node)
        else:
            self._connectAttr(items[1], '%s.input1' % node)
    
        return '%s.output' % node




# TODO
# - input/output attrs + container packaging
# - load/save
# - test everything
#VAL = 'pSphere1.t'
#MIN = [5,0,0]
#MAX = [10,10,10]
#RESULT = 'pCube1.t'

#exp = '''
#$delta  = ($VAL - $MIN)
#$range  = ($MAX - $MIN)
#$test   = ($delta/$range)
#$ratio  = 1 - exp(-1 * abs($test))
#$result = $MIN + ($ratio * $range * sign($test))
#$RESULT = if ($result<$MIN, $VAL, $result)
#'''


##e = Expression()
##print e(exp, container='fadeOutAwesomeness', variables=locals())
#e = Expression()
#e('pCube2.t = abs(pSphere1.t)')



#e = Expression(debug=False, variables=locals())
#e('$items[0].t = lerp($items[1].t, $items[2].t, $items[3].blend)', variables=locals())
#e('$items[0].s = elerp($items[1].s, $items[2].s, $items[3].blend)', variables=locals())
#print e('pCube3.t = slerp(pCube1.t, pCube2.t, pCube4.blend)')

#values = ['pCube4.choice','pCube1.wm','pCube2.wm','pCube3.wm']
#e = Expression(debug=False, variables=locals())
#e('pSphere1.t = choice($values)')
