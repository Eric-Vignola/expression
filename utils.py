

import cmd
import math
import random
import re
from collections import OrderedDict
from copy import deepcopy

import maya.cmds as mc


from pyparsing import Regex, SkipTo, Forward, Word, Combine, Literal, Optional, Group, ParseResults, ParserElement
from pyparsing import nums, alphas, alphanums, oneOf, opAssoc, infixNotation, delimitedList

ParserElement.enablePackrat() # speeds up parser

__version__ = 1.0

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
                 container   = None, 
                 variables   = None,
                 consolidate = True,
                 debug       = False):
        
        self.debug       = debug       # turns on debug prints
        self.container   = None        # name of the container to create and add all created nodes to
        self.constants   = {}          # dict of constants for replacement lookup
        self.constant    = None        # name of network node used to pack constants
        self.consolidate = consolidate # consolidate constants onto a single constants node
        self.attributes  = {}          # dict of attribute names and arguments added to the container
        self.nodes       = []          # keep track of nodes produced by the parser
        self.expression  = []          # the evaluated expression
                                       
        self.private     = {}          # private variables (declared inside the expression)
        self.variables   = {}          # user defined variables (always expanded, will never be assigned)

        # keep track of conversion helpers so they can be reused
        self._mat2quat   = {}
        self._mat2trans  = {}
        self._quat2mat   = {}
        self._quat2trans = {}


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
        

        # !!! _evalAssignOp is now done as a pre/post evaluation step
        self.parser = expression
        #self.parser = infixNotation(
            #expression, [(oneOf('='), 2, opAssoc.LEFT, self._evalAssignOp)]
        #)
        
        
        # Set the variables if specified
        self.setVariables(variables)
        
        # Set the container if need specifiec
        self.setContainer(container)



    def __call__(self, expression, name=None, variables=None, container=None, select=False):
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
        
        result = self.eval()
        
        if select:
            self.select()
            
        return result
        
        
    #def __str__(self):
        #return self.expression
        
        
    #def __repr__(self):
        #return self.expression
        
    #def __repr__(self):
        #return type(self).__name__ + '("' + '\n'.join(self.expression) + '")'


    def select(self):
        mc.select(self.nodes, r=True)



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
        """ 
        Sets user variables, only accept:
        lists, sets, tuples, string, unicode, floats and ints
        """
        if variables:
            for var in variables:
                if type(variables[var]) in [dict,OrderedDict,list,set,tuple,str,unicode,float,int]:
                    self.variables[var] = deepcopy(variables[var])
                
                
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
        
        # clear constants if node has been deleted
        if self.constant and not mc.objExists(self.constant):
            self.constant  = None
            self.constants = {}
        
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
                #stored, line = self._findVariableAssignment(line)
                stored, line = self._findDelimitedListAssignment(line)

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


                # connect both lists in parallel
                # if one list is shorter, use the last index
                # variables will be assigned, node.att connected
                if stored:
                    max0, max1 = len(stored)-1, len(solution)-1
                    maxsize = max(max0, max1) + 1
                    
                    for i in range(maxsize):
                        index0 = min(i, max0)
                        index1 = min(i, max1)
                        
                        # variable
                        if stored[index0].startswith('$'):                            
                            if self.debug:
                                print ('storing:    %s ---> %s'%(solution[index1], stored[index0]))                         
                                
                            self.private[stored[index0][1:]] = solution[index1]                    
                    
                        # node.attr
                        else:
                            if self._isMatrixAttr(solution[index1]):
                                self._connectMatrix(solution[index1], stored[index0])
                            else:
                                self._connectAttr(solution[index1], stored[index0], align_plugs=True)                         
                            



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

    def _flatten_lists(self, sequence):
        """ Flattens a deeply nested list, which can orrur when expanding variables """
        if not sequence:
            return sequence
        
        if isinstance(sequence[0], (list, tuple)):
            return self._flatten_lists(sequence[0]) + self._flatten_lists(sequence[1:])
        
        return sequence[:1] + self._flatten_lists(sequence[1:])        


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

        # !!!!!!!! BUG !!!!!!!!! #
        # Attempting to save a constant but doesn't work in the case of:
        # "1 - some_function(-1 * some_function(some_value))" because the 
        # first number is created without a connection and -1 is seen after
        # causing a the initial unconnected (but correct) attribute to be flipped
        #if self.consolidate:
            #if self.constants:
                #inv_map = {v: k for k, v in self.constants.iteritems()}
                
                ## is this a mapped constant?
                #if res in inv_map:

                    ## is it already connected to something?
                    #if not mc.listConnections(res):

                        ## if not, this is a first instance of an _evalSignOp
                        ## we can hack it and skip the multiplication step
                        #val = mc.getAttr(res)
                        #neg = -1 * val                        
                        
                        ## is the negated value already in memory?
                        #if neg in self.constants:
                            
                            ## unfortunately this will leave a gap in the multi array
                            #return self.constants[neg]
                        
                        ## replace the value in self.constants
                        #mc.setAttr(res, neg) # swap + for -
                        #self.constants[neg] = self.constants.pop(val)
                    
                        #return self.constants[neg]
                    
        if tokens[0][0] == '-':
            res = self.mult([self._long(-1, constant=True), res])
            
        return res
    
    
    def _evalAssignOp(self, tokens):
        """ Used for expression variable assignment and connection to nodes. """
        #print 'here'
        dst = tokens[0][:-2]
        src = tokens[0][-1]
        op  = tokens[0][-2]
        
        if not isinstance(dst, (tuple, list, set)):
            dst = [dst]
            
        
        if op == '=':
        
            # is the source a matrix type?
            if self._isMatrixAttr(src):
                self._connectMatrix(src, dst)        
            
            else:
                for item in dst:
                    self._connectAttr(solution[0], item, align_plugs=True)  

                           
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
                values = deepcopy(variables[var])
                
                # is it a list?
                if isinstance(values, (list, tuple, ParseResults)):
                    if isinstance(values, ParseResults):
                        values = values.asList()

                    # is the list sliced?
                    if index is not None:
                        if isinstance(index, slice):
                            values = values[index]
                        else:
                            values = [values[index]]
                            
                    # is the list all numerals?
                    if all([(isfloat(x) or isint(x)) for x in values]):
                        
                        # is it a vector?
                        if len(values) == 3:
                            return self._double3(values, constant=True)
                        
                        # is it a 16 element 4x4 matrix?
                        if len(values) == 16:
                            return self._matrix(values)

                    
                else:
                    values = [values]


                # are we overriding attributes?
                if attr:
                    for i,v in enumerate(values):
                        values[i] = '%s.%s'%(v.split('.')[0],attr)
                
         
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

    
        # if only one item queried, we're trying to figure out if this is a compound or not
        if len(query) == 1:
            if self._isCompoundAttr(query[0]):
                return self._listPlugs(query[0])[1:]
            else:
                return self._listPlugs(query[0])
    
    
        # if we have multiple entries, test if they're all compount types
        # gracefully fail if there's something weird, like a '=' for conditions
        try:
            if compound and all([self._isCompoundAttr(x) for x in query]):
                return [[self._listPlugs(x)[0]] for x in query]
        except:
            pass
    
    
        # then we flatten the entries to match the one with the most entries
        attrs = []
        for obj in query:
            attrs.append(self._listPlugs(obj))
        
        counts = [len(x) for x in attrs]
        if max(counts) == 1:
            return attrs
        
        maxi   = max(counts) - 1
        result = [[None] * maxi for _ in query]
        for i in range(len(query)):
            for j in range(maxi):
                if counts[i] == 1:
                    result[i][j] = attrs[i][0]
                else:
                    result[i][j] = attrs[i][min(counts[i], j + 1)]
    
        return result
    
    
    #def _getPlugs(self, query, compound=True):
        #"""
        #Enumerates input plugs and matches them as sets of same size.
        #ex: [pCube2.v, pCube1.t] ---> [[pCube2.v, pCube2.v, pCube2.v], [pCube1.tx, pCube1.ty, pCube1.tz]]
        #ex: compound=True  and [pCube1.t, pCube2.t] ---> [[pCube1.t], [pCube2.t]]
        #ex: compound=False and [pCube1.t, pCube2.t] ---> [[pCube1.tx, pCube1.ty, pCube1.tz], [pCube2.tx, pCube2.ty, pCube2.tz]]
        #"""
    
        #if not isinstance(query, (list, tuple, ParseResults)):
            #query = [query]
    
        #attrs = []
        #for obj in query:
            #attrs.append(self._listPlugs(obj))
            
        #counts = [len(x) for x in attrs]

        #if counts:
            #maxi = max(counts) - 1
    
            ## !!! HACK !!! #
            ## If one of the inputs is a choice node, we force compound mode
            #choice_test = False
            #try:
                #choice_test = any([mc.nodeType(q) == 'choice' for q in query])
            #except:
                #pass
    
            ## If all counts the same
            #if len(query) > 1 and (choice_test or ([counts[0]] * len(counts) == counts and compound)):
                #return [[x[0]] for x in attrs]
    
            #else:

                ## Compound mode off
                #if maxi == 0 and not compound:
                    #return attrs
    
                #result = [[None] * maxi for _ in query]

                #for i in range(len(query)):
                    #for j in range(maxi):
                        #if counts[i] == 1:
                            #result[i][j] = attrs[i][0]
                        #else:
                            #result[i][j] = attrs[i][min(counts[i], j + 1)]

                #return result
        


    def _connectMatrix(self, src='', destinations=[]):
        
        # Attempts to be nice and decompose a matrix only once
        # before attempting to connect it to multiple destinations.
        # Usually happens when connecting to a delimited list.
        
        if not isinstance(destinations, (tuple, list, set)):
            destinations = [destinations]
            

        for dst in destinations:
            
            # if destination is a matrix: direct plug
            if self._isMatrixAttr(dst):
                if self.debug:
                    print ('connecting: %s ---> %s'%(src,dst))
                    
                mc.connectAttr(src, dst, f=True)
                
            # if destination is a quat: matrixToQuat plug
            elif self._isQuatAttr(dst):
                
                # if a conversion step already made for this node, use it
                if not src in self._mat2quat:
                    self._mat2quat[src] = self.matrixToQuat(src)

                    if self.debug:
                        print ('converting: %s ---> %s'%(src, self._mat2quat[src].split('.')[0]))
                    
                
                if self.debug:
                    print ('connecting: %s ---> %s'%(self._mat2quat[src], dst))
                    
                mc.connectAttr(self._mat2quat[src], dst, f=True)
                
                
            # if destination is a transform:
            # be nice and try to figure out what to plug to what.
            elif self._isTransformAttr(dst):
                if not src in self._mat2trans:
                    self._mat2trans[src] = self.matrixDecompose(src).split('.')[0]
                    
                    if self.debug:
                        print ('converting: %s ---> %s'%(src, self._mat2trans[src]))
                    

                node, att = dst.split('.')
                
                # scale
                if att in ['scale',
                           'scaleX',
                           'scaleY',
                           'scaleZ',
                           's',
                           'sx',
                           'sy',
                           'sz']:
                        
                    self._connectAttr('%s.outputScale'%self._mat2trans[src], dst)
                    
                # rotate
                elif att in ['rotate',
                             'rotateX',
                             'rotateY',
                             'rotateZ',
                             'r',
                             'rx',
                             'ry',
                             'rz']:
                    
                    try:
                        mc.connectAttr('%s.ro'%node, '%s.inputRotateOrder'%self._mat2trans[src])  
                    except:
                        
                        # we need a new conversion plug if rotate order is already used
                        self._mat2trans.pop(src, None)
                        return self._connectMatrix(src, destinations)
                        
                                            
                        
                    self._connectAttr('%s.outputRotate'%self._mat2trans[src], dst)    
                    
                    
                    
                
                # translate
                elif att in ['translate',
                             'translateX',
                             'translateY',
                             'translateZ',
                             't',
                             'tx',
                             'ty',
                             'tz']:
                        
                    self._connectAttr('%s.outputTranslate'%self._mat2trans[src], dst)
                    
                # shear  
                elif att in ['shear']:
                    self._connectAttr('%s.outputShear'%self._mat2trans[src], dst)
                    
                    
            else:
                
                # hail mary
                try:
                    mc._connectAttr(src, dst)
                except:
                    raise Exception('Matrix object cannot be connected to: %s'%dst)
        
            
                

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
            
            
            # include any conversion nodes into the container
            if self.container:
                try:
                    unit_conversion_node = mc.listConnections(dst[i], s=True, d=False, type='unitConversion')[0]
                    if unit_conversion_node:
                        mc.container(self.container, edit=True, addNode=unit_conversion_node)
                except:
                    pass      
    
    
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
    
    
    def _isLongAttr(self, tokens):
        """ 
        Check if plug is a type int.
        """
        targets = ['long', 'bool', 'enum']
        if not isinstance(tokens, (list, set, tuple)):
            tokens = [tokens]
            
        return all([mc.getAttr(x, type=True) in targets for x in tokens])
    
    
    def _isDoubleAttr(self, tokens):
        """ 
        Check if plug is a type int, bool or enum (all valid ints).
        """
        targets = ['double']
        if not isinstance(tokens, (list, set, tuple)):
            tokens = [tokens]
            
        return all([mc.getAttr(x, type=True) in targets for x in tokens])
                    
                    
    def _isLong3Attr(self, tokens):
        """ 
        Check if plug is a type int.
        """
        targets = ['long3']
        if not isinstance(tokens, (list, set, tuple)):
            tokens = [tokens]
            
        return all([mc.getAttr(x, type=True) in targets for x in tokens])
    
    
    def _isDouble3Attr(self, tokens):
        """ 
        Check if plug is a type int, bool or enum (all valid ints).
        """
        targets = ['double3']
        if not isinstance(tokens, (list, set, tuple)):
            tokens = [tokens]
            
        return all([mc.getAttr(x, type=True) in targets for x in tokens])     
                    
                    
    def _isMatrixAttr(self, tokens):
        """ 
        Check if plug is a type matrix or not.
        """
        targets = ['matrix']
        if not isinstance(tokens, (list, set, tuple)):
            tokens = [tokens]
            
        return all([mc.getAttr(x, type=True) in targets for x in tokens])

    
    def _isQuatAttr(self, tokens):
        """ 
        Check if plug is a type quaternion or not.
        """
        targets = ['double4', 'TdataCompound']
        if not isinstance(tokens, (list, set, tuple)):
            tokens = [tokens]
            
        return all([mc.getAttr(x, type=True) in targets for x in tokens])



    def _isCompoundAttr(self, tokens):
        """
        Checks if the item is in COMPOUND_TYPES
        """
        targets = ['TdataCompound','matrix','float2','float3','double2','double3','long2','long3','short2','short3']
        
        if not isinstance(tokens, (list, set, tuple)):
            tokens = [tokens]

        return all([mc.getAttr(x, type=True) in targets for x in tokens])
    
    
    def _isTransformAttr(self, item):
        """ 
        Check if plug is on a transform node and attributes are SRTs
        """
        split = item.split('.')
        node = split[0]
        attr = '.'.join(split[1:])
        if mc.nodeType(node) == 'transform':
            if attr in ['s','sx','sy','sz','scale','scaleX','scaleY','scaleZ']:
                return True
            if attr in ['r','rx','ry','rz','rotate','rotateX','rotateY','rotateZ']:
                return True
            if attr in ['t','tx','ty','tz','translate','translateX','translateY','translateZ']:
                return True
            
            return attr == 'shear'
                
        return False



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
        variable  = Combine( '$' + 
                             Word(alphanums) + 
                             Optional('[' + Optional(oneOf('" \'')) +
                                            Optional('-') +
                                            Optional(Word(nums)) +
                                            Optional(':') +
                                            Optional('-') +
                                            Optional(Word(nums)) +
                                            Optional(':') +
                                            Optional('-') +
                                            Optional(Word(alphanums)) +
                                            Optional(oneOf('" \'')) +']') + 
                             Optional('.'+Word(alphanums) ) 
                             ) # $pi, $list_of_nodes[:-2].t, $MESH.tx         
                             
                             
        node_attr = Combine(Optional('$') + Word(alphanums + '_:') + '.' + Word(alphanums + '_[].') )
        delim = Group(delimitedList(variable | node_attr, delim=',') + '=')
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
                    delim = delim[0][0][:-1]

                    # process the list
                    result = []
                    for i, item in enumerate(delim):
                        
                        # is this a node.attr?
                        if not item.startswith('$'):
                            split = item.split('.')
                            node = split[0]
                            attr = '.'.join(split[1:])
                            
                            if not mc.objExists(item):
                                raise Exception('Assigned object %s does not exist.'%item)
                            
                            elif not mc.attributeQuery(attr, node=node, exists=True):
                                raise Exception('Assigned object attribute %s does not exist.'%item)
                            
                            else:
                                result.append(item)
                            
                            
                        # process variable
                        else:
                            var, index, attr = self._splitVariable(item)

                            # is this a user variable? (expand it)
                            if var in self.variables:
                                obj = self.variables[var]

                                # is there an index, slice or dict key
                                if index:
                                    obj = obj[index]
                                    
                                # no key but obj is dict: expand all values
                                elif isinstance(obj, (dict, OrderedDict)):
                                    obj = obj.values()
                                    
                                # make the object a list for the next step
                                if isinstance(obj, tuple):
                                    obj = list(obj)
                                    
                                elif not isinstance(obj, list):
                                    obj = [obj]
                                    
                                # override any attribute?
                                if attr:
                                    for j,v in enumerate(obj):
                                        obj[j] = '%s.%s'%(v.split('.')[0] ,attr)
                                    
                                result.append(obj)
                                
                                
                            # allow attribute override for private vars
                            elif attr and var in self.private:
                                obj = self.private[var]
                                if not isinstance(obj, list):
                                    obj = [obj]                                
                                
                                for j,v in enumerate(obj):
                                    obj[j] = '%s.%s'%(v.split('.')[0] ,attr)
                                
                                result.append(obj)                                
                                        
                                        
                            # assume variable is private (don't expand)
                            else:
                                result.append(item) # keep the $variable as is                            
                                    
                                    

                    result = self._flatten_lists(result)
                    return result, line
                
        
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
            
            # is it a dict key?
            key = re.findall(r'["\'](.*?)["\']', var) # look for text between " and '
            if key:
                var = var.split('[')[0]
                var_slice = key[0]
                
            # is it a list indes, or a slice?
            else:
            
                var_slice = var[var.find('[')+1:var.find(']')]
                var = var.split('[')[0]
                if ':' in var_slice:
                    var_slice = slice(*[{True: lambda n: None, False: int}[x == ''](x) for x in (var_slice.split(':') + ['', '', ''])[:3]])
                else:
                    try:
                        var_slice = int(var_slice)
                    except:
                        pass

        return (var, var_slice, attr)
        
    
    
    

    # ----------------------------- GENERAL NODES ------------------------------ #
    
    # TODO: figure out scheme to avoid unit conversions and create proper node type
    #       betwen animCurveTU, animCurveTA and animCurveTL
    @parsedcommand
    def frame(self, tokens):
        """ 
        frame()
        
            Outputs "current frame" via an infinite linear motion curve.
        
            Examples
            --------
            >>> frame() # returns a current time slider value.
        """
    
        if tokens:
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
    def noise(self, tokens):
        """ 
        noise(<input>)
        
            Creates a pseudo random function via perlin noise.
        
            Examples
            --------
            >>> noise(pCube1.tx) # Applies noise to pCube1's translateX value.
            >>> noise(pCube1.t)  # Applies noise to pCube1's [tx, ty, tz] values.
        """
    
        # Handle single value or vector
        tokens = self._getPlugs(tokens, compound=False)
        
        exp = Expression(container='noise1', debug=self.debug)
        results = []
        for i in range(len(tokens[0])):
            plug = tokens[0][i]
    
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
    def lerp(self, tokens):
        """ 
        lerp(<input>, <input>, <weight>)
        
            Linear interpolation between two inputs and a weight value.
        
            Examples
            --------
            >>> lerp(pCube1.tx, pCube2.tx, pCube3.weight)  # Computes the magnitude of [tx, ty, tz].
        """
        if len(tokens) != 3:
            raise Exception('lerp requires 3 inputs, given: %s' % tokens)
    
        exp = Expression(container='lerp1', 
                         variables=locals(), 
                         debug=self.debug)
        
        code = '''
        $tokens[0] + $tokens[2] * ($tokens[1]-$tokens[0])
        '''
        result = exp(code)
        self.nodes.extend(exp.getNodes())
        
        return result
    
    
    @parsedcommand
    def elerp(self, tokens):
        """ 
        elerp(<input>, <input>, <weight>)
        
            Exponent linear interpolation between two inputs and a weight value.
        
            Examples
            --------
            >>> elerp(pCube1.tx, pCube2.tx, pCube3.weight)  # Computes the magnitude of [tx, ty, tz].
        """
        if len(tokens) != 3:
            raise Exception('lerp requires 3 inputs, given: %s' % tokens)
    
    
        exp = Expression(container='elerp1', 
                         variables=locals(), 
                         debug=self.debug)
        
        node = exp('$tokens[0]**rev($tokens[2]) * $tokens[1]**$tokens[2]')[0]
        self.nodes.extend(exp.getNodes())
        return node 
    
    @parsedcommand
    def slerp(self, tokens):
        """ 
        slerp(<input>, <input>, <weight>)
        
            Spherical linear interpolation between two inputs and a weight value.
        
            Examples
            --------
            >>> slerp(pCube1.tx, pCube2.tx, pCube3.weight)  # Computes the magnitude of [tx, ty, tz].
        """
        if len(tokens) != 3:
            raise Exception('slerp requires 3 inputs, given: %s' % tokens)
    
        v0 = tokens[0]
        v1 = tokens[1]
        blend = tokens[2]
    
        exp = Expression(container='slerp1', variables=locals(), debug=self.debug)   
        exp('$angle = acos(dot(unit($v0), unit($v1)))')
        node = exp('(($v0*sin(rev($blend)*$angle)) + ($v1*sin($blend*$angle)))/sin($angle)')
        self.nodes.extend(exp.getNodes())
        return node         
        
        
    

    @parsedcommand
    def mag(self, tokens):
        """ 
        mag(<input>)
        
            Returns the magnitude of a vector.
        
            Examples
            --------
            >>> mag(pCube1.t)  # Computes the magnitude of [tx, ty, tz].
        """
        if len(tokens) != 1:
            raise Exception('mag requires 1 input, given: %s' % tokens)
    
        node = self._createNode('distanceBetween', name='magnitude1', ss=True)
        self._connectAttr(tokens[0], '%s.point1' % node)
    
        return '%s.distance' % node


    @parsedcommand
    def cond(self, tokens):
        """ 
        cond(<input> <op> <input>, <input if true>, <input if false>)
        
            Creates condition node to solve "if" statements.
        
            Examples
            --------
            >>> cond(pCube1.t > pCube2.t, 0, pCube3.t)
            >>> cond(pCube1.rx < 45, pCube1.rx, 45) # outputs pCube1.rx's value with a maximum of 45
        """

        if len(tokens) != 5:
            raise Exception('cond() needs 5 items: [a,cond_op,b,val if true,val if false]. Given: %s' % tokens)
    
        A, op, B, true, false = self._getPlugs(tokens)

        
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
    def clamp(self, tokens):
        """ 
        clamp(<input>, <input min>, <input max>)
        
            Clamps values between a min and a max.
        
            Examples
            --------
            >>> clamp(pCube1.ty, 0, pCube2.ty) # clamps pCube1.ty value between 0 and pCube2.ty
            >>> clamp(pCube1.t, -1, 1) # clamps [tx, ty, tz] of pCube1 between -1 and 1
        """
    
        if len(tokens) != 3:
            raise Exception('clamp() requires 3 inputs, given: %s' % tokens)
    
        # all inputs differ, use a clamp node
        if tokens[0] != tokens[1] and tokens[0] != tokens[2]:
            node = self._createNode('clamp', ss=True)
            mc.setAttr('%s.renderPassMode' % node, 0)
            self._connectAttr(tokens[1], '%s.min' % node)
            self._connectAttr(tokens[2], '%s.max' % node)
            self._connectAttr(tokens[0], '%s.input' % node)
    
            counts = self._getPlugs(tokens, compound=False)
            if all(len(x) == 1 for x in counts):
                return ['%s.outputR' % node]
    
            return ['%s.output' % node]
    
        # shared input, use 2 conditions because clamp can't deal with it
        # max(min(my_value, max_value), min_value)
        exp = Expression(container='clamp1', 
                         debug=self.debug)
        
        MIN = exp.cond([tokens[0], '<', tokens[2], tokens[0], tokens[2]])
        result = exp.cond([MIN, '>', tokens[1], MIN, tokens[1]])   
        
        self.nodes.extend(exp.getNodes())
        return result
    
    
    
    
    # ----------------------------- NUMERICAL NODES ------------------------------ #  
    
    def _number(self, value=0, at='double', name='number1', constant=False):
        """
        Creates a single numeric value using a network node (default float).
        """
        
        # return repacked constant
        if constant and self.consolidate:
            
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
        if constant and self.consolidate:        
        
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
    
    
    def _matrix(self, values=None, name='matrix1'):
        """
        Creates a matrix out of the 16 given values
        """           
        M = self._createNode('fourByFourMatrix', ss=True)
        index = 0
        for i in range(4):
            for j in range(4):
                plug = '%s.in%s%s' % (M, i, j)
                mc.setAttr(plug, values[index])
                
                index+=1
                    
        return '%s.output' % M

                            
                            
        
        
    
    
    # -------------------------- PEMDAS MATH FUNCTIONS --------------------------- #

    def _multiplyDivide(self, op, tokens):
        """
        PEMDAS multiply/divide operations
        """
        mat0 = self._isMatrixAttr(tokens[0])
        mat1 = self._isMatrixAttr(tokens[1])
    
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
                result = exp('floor(%s/%s)' % (tokens[0], tokens[1]))[0]
                
                self.nodes.extend(exp.getNodes())
                return result                
            
            # modulo
            elif op == '%':
                exp = Expression(container='modulo1' ,debug=self.debug)
                result = exp('%s - floor(%s/%s)*%s' % (tokens[0], tokens[0], tokens[1], tokens[1]))[0]
                
                self.nodes.extend(exp.getNodes())                
                return result

            
            else:
                raise Exception('unsupported operator: %s' % op)
    
            self._connectAttr(tokens[0], '%s.input1' % node)
            self._connectAttr(tokens[1], '%s.input2' % node)
    
            # Force single output if both inputs are single numerics
            if all(not self._isCompoundAttr(x) for x in tokens):
                return '%s.outputX' % node
    
            return '%s.output' % node
    
        else:
    
            # is this a mat * mat
            if mat0 and mat1:
                return self.matrixMult(tokens)
    
            # is this a mat * p
            else:
                return self.pointMatrixProduct(tokens)
        
        
        
    def _plusMinusAverage(self, op, tokens):
        """
        PEMDAS plus, minus, average operations
        """
    
        mat0 = self._isMatrixAttr(tokens[0])
        mat1 = self._isMatrixAttr(tokens[1])
    
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
            if not any(self._isCompoundAttr(x) for x in tokens):
                for i, obj in enumerate(tokens):
                    self._connectAttr(obj, '%s.input1D[%s]' % (node, i))
    
                return '%s.output1D' % node
    
            # Connect
            for i, obj in enumerate(tokens):
                self._connectAttr(obj, '%s.input3D[%s]' % (node, i))
    
            return '%s.output3D' % node
    
    
        else:
            
            if op == '+':
                return self.matrixAdd(tokens)
            
            elif op == 'avg':
                return self.matrixWeightedAdd(tokens)
                
            else:
                raise Exception('Unsupported %s operator used on matrices.'%op)
    
    
    @parsedcommand
    def power(self, tokens):
        """
        Return x raised to the power y
        """
        return self._multiplyDivide('**', tokens)
                
    @parsedcommand            
    def mult(self, tokens):
        """
        Multiplies two or more items
        """
        return self._multiplyDivide('*', tokens)
                
    @parsedcommand            
    def div(self, tokens):
        """
        Divides two or more items
        """        
        return self._multiplyDivide('/', tokens)
    
    @parsedcommand
    def add(self, tokens):
        """
        Adds two or more items
        """
        return self._plusMinusAverage('+', tokens)   
        
    @parsedcommand    
    def sub(self, tokens):
        """
        Subtracts two or more items
        """
        return self._plusMinusAverage('-', tokens)   
            
    @parsedcommand        
    def avg(self, tokens):
        """
        Averages two or more items
        """
        return self._plusMinusAverage('avg', tokens)
    


    
    # -------------------------- COMMON MATH FUNCTIONS --------------------------- #
    
    @parsedcommand
    def inv(self, tokens):
        """ 
        inv(<input>)
        
            Creates a network to yields an inverse (0.0-x) mirror operation.
        
            Examples
            --------
            >>> inv(pCube1.t) # returns -pCube1.t
            >>> inv(pCube1.wm) # returns matrixInverse(pCube1.wm)
        """
        if len(tokens) != 1:
            raise Exception('inverse() requires 1 input, given: %s' % tokens)
    
        if self._isMatrixAttr(tokens[0]):
            return self.matrixInverse(tokens[0])
        else:
            
            exp = Expression(container='inv1', 
                             debug=self.debug)

            result = exp('0 - %s'%tokens[0])
            
            self.nodes.extend(exp.getNodes())  
            return result
    
    
    @parsedcommand
    def rev(self, tokens):
        """ 
        rev(<input>)
        
            Creates a reverse node to do a (1.0-x) operation.
        
            Examples
            --------
            >>> rev(pCube1.t)
        """
        if len(tokens) != 1:
            raise Exception('rev() requires 1 input, given: %s' % tokens)
    
        node = self._createNode('reverse', ss=True)
        self._connectAttr(tokens[0], '%s.input' % node)
    
        return '%s.output' % node    
    
        
    @parsedcommand
    def sum(self, tokens):
        """ 
        sum(<input>, <input>, <input>, ...)
        
            Single node operation to sum all items in the list.
        
            Examples
            --------
            >>> sum(pCube1.t, pCube2.t, pCube3.t, pCube4.t)
        """
    
        if len(tokens) < 2:
            raise Exception('sum() requires minimum 2 inputs, given: %s' % tokens)
    
        return self.add(tokens)        
        
    @parsedcommand    
    def int(self, tokens):
        """ 
        int(<input>)
        
            Turns a float value(s) into an int.
        
            Examples
            --------
            >>> int(pCube1.t)
            >>> int(pCube1.tx)
        """
        
        exp = Expression(container='int1', 
                         debug=self.debug)
            
        if len(tokens) != 1:
            raise Exception('int() requires 1 input, given: %s' % tokens)
    
        node = None
        if len(self._getPlugs(tokens[0])[0]) > 1:
            node = exp._long3()
        else:
            node = exp._long()
    
        obj = tokens[0]
        exp('$f = 0.4999999')
        exp('$true  = $obj - $f', variables=locals())
        exp('$false = $obj + $f', variables=locals())
        result = exp('if ($obj > 0, $true, $false)', variables=locals())
        
        self.nodes.extend(exp.getNodes())  
        self._connectAttr(result[0],node)
        
        return node
    
        
    
    
    @parsedcommand
    def max(self, tokens):
        """ 
        max(<input>, <input>, <input>, ...)
        
            Returns the highest value in the list of inputs.
        
            Examples
            --------
            >>> max(pCube1.t, pCube2.t, pCube3.t, pCube4.t)
        """
    
        if len(tokens) < 2:
            raise Exception('max() requires minimum 2 inputs, given: %s' % tokens)
    
        exp = Expression(container='max1', 
                         debug=self.debug)
        
        ret = tokens[0]
        for obj in tokens[1:]:
            ret = exp.cond([ret, '>', obj, ret, obj])
            
        self.nodes.extend(exp.getNodes())  
    
        return ret
    
    @parsedcommand
    def min(self, tokens):
        """ 
        min(<input>, <input>, <input>, ...)
        
            Returns the lowest value in the list of inputs.
        
            Examples
            --------
            >>> min(pCube1.t, pCube2.t, pCube3.t, pCube4.t)
        """
    
        if len(tokens) < 2:
            raise Exception('min() requires minimum 2 inputs, given: %s' % tokens)
    
        exp = Expression(container='min1', 
                         debug=self.debug)
        
        ret = tokens[0]
        for obj in tokens[1:]:
            ret = exp.cond([ret, '<', obj, ret, obj])
    
        self.nodes.extend(exp.getNodes())  
        return ret
    
    
    @parsedcommand
    def exp(self, tokens):
        """ 
        exp(<input>)
        
            Return e raised to the power x, 
            where e is the base of natural logarithms (2.718281...)
        
            Examples
            --------
            >>> exp(pCube1.tx)
        """
        return self.power([self._double(math.e, constant=True), tokens[0]])
    
    
    @parsedcommand
    def sign(self, tokens):
        """ 
        sign(<input>)
        
            Returns -1 for values < 0. +1 for values >= 0.
        
            Examples
            --------
            >>> sign(pCube1.t)
            >>> sign(pCube1.tx)
        """
        if len(tokens) != 1:
            raise Exception('sign() requires 1 input, given: %s' % tokens)
    
        return self.cond([tokens[0], 
                          '<', 
                          self._long(0, constant=True), 
                          self._long(-1, constant=True), 
                          self._long(1, constant=True)])   
    
    
    @parsedcommand
    def sqrt(self, tokens):
        """ 
        sqrt(<input>)
        
            Returns the square root of a value. (sam as doing x ** 0.5)
        
            Examples
            --------
            >>> sqrt(pCube1.t)
            >>> sqrt(pCube1.tx)
        """
        if len(tokens) != 1:
            raise Exception('sign() requires 1 input, given: %s' % tokens)
    
        return self.power([tokens[0], 
                          self._double(0.5, constant=True)])        
    
    
    @parsedcommand
    def floor(self, tokens):
        """ 
        floor(<input>)
        
            Returns the floor value of the input.
        
            Examples
            --------
            >>> floor(pCube1.t)
            >>> floor(pCube1.tx)
        """
        if len(tokens) != 1:
            raise Exception('floor() requires 1 input, given: %s' % tokens)
    
        
        exp = Expression(container='floor1', 
                         debug=self.debug)

        result = exp('%s - 0.4999999'%tokens[0])
        if len(self._getPlugs(tokens)) > 1:
            node = exp._long3()
        else:
            node = exp._long()
    
        self._connectAttr(result[0], node)
        self.nodes.extend(exp.getNodes())  
        
        return node             
        

    
    
    @parsedcommand
    def ceil(self, tokens):
        """ 
        ceil(<input>)
        
            Returns the ceil value of the input.
        
            Examples
            --------
            >>> ceil(pCube1.t)
            >>> ceil(pCube1.tx)
        """
    
        exp = Expression(container='ceil1', 
                         debug=self.debug)

        result = exp('0.4999999 + %s'%tokens[0])
        if len(self._getPlugs(tokens)) > 1:
            node = exp._long3()
        else:
            node = exp._long()
    
        self._connectAttr(result[0], node)
        self.nodes.extend(exp.getNodes())  
        
        return node       
        
    
    @parsedcommand    
    def dist(self, tokens):
        """ 
        dist(<input>, <input>)
        
            Creates a distanceBetween node to find distance between points or matrices.
        
            Examples
            --------
            >>> dist(pCube1.t, pCube2.t)
            >>> dist(pCube1.wm, pCube2.wm)
        """

        if len(tokens) != 2:
            raise Exception('clamp requires 2 inputs, given: %s' % tokens)
    
        node = self._createNode('distanceBetween', ss=True)
    
        if self._isMatrixAttr(tokens[0]):
            self._connectAttr(tokens[0], '%s.inMatrix1' % node)
        else:
            self._connectAttr(tokens[0], '%s.point1' % node)
    
        if self._isMatrixAttr(tokens[1]):
            self._connectAttr(tokens[1], '%s.inMatrix2' % node)
        else:
            self._connectAttr(tokens[1], '%s.point2' % node)
    
        return '%s.distance' % node    
        

    @parsedcommand
    def abs(self, tokens):
        """ 
        abs(<input>)
        
            Outputs the absolute value of a float or vector.
        
            Examples
            --------
            >>> abs(pCube1.t)
            >>> abs(pCube1.tx)
        """
    
        tokens = self._getPlugs(tokens, compound=False)
    
        exp  = Expression(container='abs1', 
                          debug=self.debug)
        
        result = []
        for item in tokens:
            result.append(exp('if($item < 0, -1 * $item, $item)', variables=locals()))
            
        result = self._flatten_lists(result)
    
        if len(result) > 1:
            result = exp('vector($result)\n', variables=locals())
    
        self.nodes.extend(exp.getNodes())  
        
        return result
        
        
        
    @parsedcommand   
    def choice(self, tokens):
        """ 
        choice(<selector>, <input>, <input>, ...)
        
            Creates a choice node out of inputs.
            If selector is None, nothing will be set.
        
            Examples
            --------
            >>> choice(pCube1.someEnum, pCube2.wm, pCube3.wm)
            >>> choice(None, pCube2.wm, pCube3.wm) # leaves selector unplugged.
        """

        if len(tokens) < 2:
            raise Exception('choice requires minimum 2 inputs, given: %s' % tokens)
    
        # create choice node
        node = self._createNode('choice', ss=True)
        
        # plug selector
        if not tokens[0] in [None, 'None']:
            
            if self._isLongAttr(tokens[0]):
                self._connectAttr(tokens[0], '%s.selector' % node)
                
            else:
                integer = self('int(%s)'%tokens[0])[0]
                self._connectAttr(integer, '%s.selector' % node)
    
        # plug inputs
        for item in tokens[1:]:
            self._connectAttr(item, '%s.input' % node)
    
        return '%s.output' % node
    
    
    @parsedcommand
    def vector(self, tokens):
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
            if not tokens[i] in [None, 'None']:
                self._connectAttr(tokens[i], xyz)
    
        return node    
    
    
    
    
    # -------------------------- ALGEBRA MATH FUNCTIONS -------------------------- #
    
    # TODO: support Matrix dot product?
    @parsedcommand
    def _vectorProduct(self, name, op, normalize, tokens):
        """
        Vector Product wrapper
        """
        if len(tokens) != 2:
            raise Exception('%s requires 2 inputs, given: %s' % (name, tokens))
    
        node = self._createNode('vectorProduct', ss=True)
        mc.setAttr('%s.operation' % node, op)
        mc.setAttr('%s.normalizeOutput' % node, normalize)
    
        self._connectAttr(tokens[0], '%s.input1' % node)
        self._connectAttr(tokens[1], '%s.input2' % node)
    
        return '%s.output' % node            
    
    
    @parsedcommand
    def dot(self, tokens):
        """ 
        dot(<input>, <input>)
        
            Uses a vectorProduct to do a dot product between two vector inputs.
        
            Examples
            --------
            >>> dot(pCube1.t, pCube2.t)
        """
        return self._vectorProduct('dot', 1, 0, tokens)
        
        
    @parsedcommand
    def dotNormalized(self, tokens):
        """ 
        dotNormalized(<input>, <input>)
        
            Uses a normalized vectorProduct to do a dot product between two vector inputs.
        
            Examples
            --------
            >>> dotNormalized(pCube1.t, pCube2.t)
        """
        return self._vectorProduct('dotNormalized', 1, 1, tokens)
    
    
    @parsedcommand
    def cross(self, tokens):
        """ 
        cross(<input>, <input>)
        
            Uses a vectorProduct to do a cross product between two vector inputs.
        
            Examples
            --------
            >>> cross(pCube1.t, pCube2.t)
        """
        return self._vectorProduct('cross', 2, 0, tokens)
    
    
    @parsedcommand
    def crossNormalized(self, tokens):
        """ 
        crossNormalized(<input>, <input>)
        
            Uses a normalized vectorProduct to do a cross product between two vector inputs.
        
            Examples
            --------
            >>> crossNormalized(pCube1.t, pCube2.t)
        """
    
        return self._vectorProduct('crossNormalized', 2, 1, tokens)
    
    
    @parsedcommand
    def unit(self, tokens):
        """ 
        unit(<input>)
        
            Creates a network that yields a unit vector.
        
            Examples
            --------
            >>> unit(pCube1.t)
        """
    
        if len(tokens) != 1:
            raise Exception('unit() requires 1 input, given: %s' % tokens)
    
        
        exp  = Expression(container='unit1', 
                          variables={'node':tokens[0]},
                          debug=self.debug)
        code = '''
        
        # divide input by magnitude
        $mag = mag($node)
        $div = $node/$mag
        
        # this will set node to has no effect when div by zero
        $div.operation = if($mag==0, 0, 2)
        
        # final return
        if($mag==0, 0, $div)
        '''
        result = exp(code)
        self.nodes.extend(exp.getNodes())    
    
        return result     
    
    
    
    # ------------------------- TRIGONOMETRIC FUNCTIONS -------------------------- # 
    def _trigonometry(self, tokens, x, y, modulo=None, container=None):
        """
        Sets up a ramapValue node for sine/cosine trigonometric functions.
        """
        tokens = self._getPlugs(tokens, compound=False)
        
        exp = Expression(container=container, debug=self.debug)
        results = []
        for i in range(len(tokens)):
            plug = tokens[i]

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
        
        
        
    def _trigonometry2(self, tokens, name, convert_value, output):
        
        
        tokens = self._getPlugs(tokens, compound=False)
        if not len(tokens) in [1,3]:
            raise Exception('trigonometric functions ony supports 1 or 3 plugs')
        
        nodes = []
        exp = Expression(container=name, debug=self.debug)
                   
        for token in tokens:
            node    = exp._createNode('eulerToQuat', ss=False)
            result  = exp('%s.inputRotateX = %s*%s'%(node, token, convert_value))[0]
            nodes.append('%s.%s'%(node,output))
        
        # make result a vector if there are 3 outputs
        result = nodes[0]
        if len(nodes) == 3:
            vec = exp._double3()
            for i, xyz in enumerate(['X', 'Y', 'Z']):
                mc.connectAttr(nodes[i], '%s%s' % (vec, xyz), f=True)
                
            result = vec
 
 
        self.nodes.extend(exp.getNodes())           
        return result      
        
        
    @parsedcommand    
    def degrees(self, tokens):
        """
        degrees(<input>)
        
            Converts incomming values from radians to degrees.
            (obj in radians * 57.29577951)
        
            Examples
            --------
            >>> degrees(radians(pCube1.rx)) # returns a network which converts rotationX to radians and back to degrees.
            >>> degrees(radians(pCube1.r))  # returns a network which converts [rx, ry, rz] to radians and back to degrees.
        """
        exp = Expression(container='degrees1', debug=self.debug)
        result = exp('%s * %s' % (tokens[0], (180./math.pi) ))[0]
        
        self.nodes.extend(exp.getNodes())
        return result


    @parsedcommand
    def radians(self, tokens):
        """ 
        radians(<input>)
        
            Converts incomming values from degrees to radians.
            (input in degrees * 0.017453292)
        
            Examples
            --------
            >>> radians(pCube1.rx) # returns a network which converts rotationX to radians.
            >>> radians(pCube1.r)  # returns a network which converts [rx, ry, rz] to radians.
        """
        exp = Expression(container='radians1', debug=self.debug)
        result = exp('%s * %s' % (tokens[0], (math.pi/180.) ))[0]
        
        self.nodes.extend(exp.getNodes()) 
        return result    


    # TODO: add built in start, stop remap values
    @parsedcommand
    def easeIn(self, tokens):
        """ 
        easeIn(<input>)
        
            Creates an easeIn "tween" function.
        
            Examples
            --------
            >>> easeIn(pCube1.tx) # returns a network which tweens pCube1's translateX value.
            >>> easeIn(pCube1.t)  # returns a network which tweens pCube1's [tx, ty, tz] values.
        """
        return self._trigonometry(tokens, x=[0, 1], y=[0, 1], container='easeIn')


    # TODO: add built in start, stop remap values
    @parsedcommand
    def easeOut(self, tokens):
        """ 
        easeOut(<input>)
        
            Creates an easeIn "tween" function.
        
            Examples
            --------
            >>> easeOut(pCube1.tx) # returns a network which tweens pCube1's translateX value.
            >>> easeOut(pCube1.t)  # returns a network which tweens pCube1's [tx, ty, tz] values.
        """
        return self._trigonometry(tokens, x=[1, 0], y=[0, 1], container='easeOut')



    @parsedcommand
    def sin(self, tokens):
        
        return self._trigonometry2(tokens, 
                                  name='sin1', 
                                  convert_value=(360./math.pi), 
                                  output='outputQuat.outputQuatX')
    
    @parsedcommand
    def sind(self, tokens):
        
        return self._trigonometry2(tokens, 
                                  name='sind1', 
                                  convert_value=2, 
                                  output='outputQuat.outputQuatX')
    
        
    @parsedcommand
    def cos(self, tokens):
        
        return self._trigonometry2(tokens, 
                                  name='cos1', 
                                  convert_value=(360./math.pi), 
                                  output='outputQuat.outputQuatW')
    
    @parsedcommand
    def cosd(self, tokens):
        
        return self._trigonometry2(tokens, 
                                  name='cosd1', 
                                  convert_value=2,
                                  output='outputQuat.outputQuatW')    
    
    
    #@parsedcommand
    #def sin(self, tokens):
        #""" 
        #sin(<input>)
        
            #Creates a sine function (in radians).
        
            #Examples
            #--------
            #>>> sin(pCube1.tx) # returns a network which passes pCube1's translateX into a sine function.
            #>>> sin(pCube1.t)  # returns a network which passes pCube1's [tx, ty, tz] into a sine functions.
        #"""
        #x = [(-5 * math.pi / 2), (5 * math.pi / 2), (-3 * math.pi / 2), (-1 * math.pi / 2), (math.pi / 2), (3 * math.pi / 2)]
        #y = [-1, 1, 1, -1, 1, -1]
        #return self._trigonometry(tokens, x, y, modulo=2 * math.pi, container='sin1')


    #@parsedcommand
    #def sind(self, tokens):
        #""" 
        #sind(<input>)
        
            #Creates a sine function (in degrees).
        
            #Examples
            #--------
            #>>> sind(pCube1.tx) # returns a network which passes pCube1's translateX into a sine function.
            #>>> sind(pCube1.t)  # returns a network which passes pCube1's [tx, ty, tz] into a sine functions.
        #"""
        #x = [(-5 * 180. / 2), (5 * 180. / 2), (-3 * 180. / 2), (-1 * 180. / 2), (180. / 2), (3 * 180. / 2)]
        #y = [-1, 1, 1, -1, 1, -1]        
        #return self._trigonometry(tokens, x, y, modulo=2 * 180., container='sind1')

    #@parsedcommand
    #def cos(self, tokens):
        #""" 
        #cos(<input>)
        
            #Creates a cosine function (in radians).
        
            #Examples
            #--------
            #>>> cos(pCube1.tx) # returns a network which passes pCube1's translateX into a cosine function.
            #>>> cos(pCube1.t)  # returns a network which passes pCube1's [tx, ty, tz] into a cosine functions.
        #"""
    
        #x = [(-2 * math.pi), (2 * math.pi), (-1 * math.pi), 0, math.pi]
        #y = [1, 1, -1, 1, -1]
        #return self._trigonometry(tokens, x, y, modulo=2 * math.pi, container='cos1')    
    
    #@parsedcommand
    #def cosd(self, tokens):
        #""" 
        #cosd(<input>)
        
            #Creates a cosine function (in degrees).
        
            #Examples
            #--------
            #>>> cosd(pCube1.tx) # returns a network which passes pCube1's translateX into a cosine function.
            #>>> cosd(pCube1.t)  # returns a network which passes pCube1's [tx, ty, tz] into a cosine functions.
        #"""
        #x = [(-2 * 180.), (2 * 180.), (-1 * 180.), 0, 180.]
        #y = [1, 1, -1, 1, -1]
        #return self._trigonometry(tokens, x, y, modulo=2 * 180., container='cosd1')   
    


    @parsedcommand
    def asind(self, tokens):
        """ 
        asin(<input>)
        
            Calculates an arc sine function (in degrees).
        
            Examples
            --------
            >>> asin(pCube1.tx) # returns a network which passes pCube1's translateX into an arc sine function.
            >>> asin(pCube1.t)  # returns a network which passes pCube1's [tx, ty, tz] into an arc sine  functions.
        """        
        
        tokens = self._getPlugs(tokens, compound=False)
        if not len(tokens) in [1,3]:
            raise Exception('trigonometric functions ony supports 1 or 3 plugs')
        
        nodes = []
        exp = Expression(container='asind1', debug=self.debug)
                   
        for token in tokens:
            node = exp._createNode('angleBetween', ss=False)
            mc.setAttr('%s.vector1'%node,0,0,0)
            mc.setAttr('%s.vector2'%node,0,0,0)
            
            angle = '%s.axisAngle.angle'%node # TODO: PUT THIS BACK IN EXPRESSION (FX ATTR EXPANSION BUG)
            
            
            code = '''
            
            # see asin under https://www.chadvernon.com/blog/trig-maya/
            
            $b = (1.0 - $token*$token)**0.5
            $node.vector1X = $b
            $node.vector1Y = $token
            
            $node.vector2X = if (abs($token) == 1.0, 1.0, $b)
            
            if ($token < 0, -$angle, $angle)
            '''
            
            result  = exp(code, variables=locals())[0]
            nodes.append(result)
        
        # make result a vector if there are 3 outputs
        result = nodes[0]
        if len(nodes) == 3:
            vec = exp._double3()
            for i, xyz in enumerate(['X', 'Y', 'Z']):
                mc.connectAttr(nodes[i], '%s%s' % (vec, xyz), f=True)
                
            result = vec
 
        self.nodes.extend(exp.getNodes())           
        return result
    
    

    @parsedcommand
    def asin(self, tokens):
        """ 
        asin(<input>)
        
            Calculates an arc sine function (in radians).
        
            Examples
            --------
            >>> asin(pCube1.tx) # returns a network which passes pCube1's translateX into an arc sine approximation function.
            >>> asin(pCube1.t)  # returns a network which passes pCube1's [tx, ty, tz] into an arc sine approximation functions.
        """
        
    
        # pack in it's own container to reduce the clutter
        exp = Expression(container='asin1', 
                         debug=self.debug)
        
        result = exp('radians(asind(%s))' % tokens[0])
        self.nodes.extend(exp.getNodes())
        
        return result    

      
    @parsedcommand
    def acosd(self, tokens):
        """ 
        acosd(<input>)
        
            Calculates an arc cosine function (in degrees).
        
            Examples
            --------
            >>> acosd(pCube1.tx) # returns a network which passes pCube1's translateX into an arc cosine function.
            >>> acosd(pCube1.t)  # returns a network which passes pCube1's [tx, ty, tz] into an arc cosine functions.
        """
        
        tokens = self._getPlugs(tokens, compound=False)
        if not len(tokens) in [1,3]:
            raise Exception('trigonometric functions ony supports 1 or 3 plugs')
        
        nodes = []
        exp = Expression(container='acosd1', debug=self.debug)
                   
        for token in tokens:
            node = exp._createNode('angleBetween', ss=False)
            mc.setAttr('%s.vector1'%node,0,0,0)
            mc.setAttr('%s.vector2'%node,0,0,0)
            
            angle = '%s.axisAngle.angle'%node # TODO: PUT THIS BACK IN EXPRESSION (FX ATTR EXPANSION BUG)
            
            
            code = '''
            
            # see acos under https://www.chadvernon.com/blog/trig-maya/
            
            $b = (1.0 - $token*$token)**0.5
            $node.vector1Y = $b
            $node.vector1X = $token
            
            $node.vector2X = if (abs($token) == 1.0, 1.0, $b)
            
            if ($token < 0, -$angle, $angle)
            '''
            
            result  = exp(code, variables=locals())[0]
            nodes.append(result)
        
        # make result a vector if there are 3 outputs
        result = nodes[0]
        if len(nodes) == 3:
            vec = exp._double3()
            for i, xyz in enumerate(['X', 'Y', 'Z']):
                mc.connectAttr(nodes[i], '%s%s' % (vec, xyz), f=True)
                
            result = vec
 
        self.nodes.extend(exp.getNodes())           
        return result
    
    
    @parsedcommand
    def acos(self, tokens):
        """ 
        acos(<input>)
        
            Calculates an arc cosine function (in radians).
        
            Examples
            --------
            >>> asin(pCube1.tx) # returns a network which passes pCube1's translateX into an arc cosine approximation function.
            >>> asin(pCube1.t)  # returns a network which passes pCube1's [tx, ty, tz] into an arc cosine approximation functions.
        """
        
    
        # pack in it's own container to reduce the clutter
        exp = Expression(container='acos1', 
                         debug=self.debug)
        
        result = exp('radians(acosd(%s))' % tokens[0])
        self.nodes.extend(exp.getNodes())
        
        return result            
    
    

    @parsedcommand
    def tan(self, tokens):
        """ 
        tan(<input>)
        
            Approximates a tan function (in radians).
        
            Examples
            --------
            >>> tan(pCube1.tx) # returns a network which passes pCube1's translateX into a tan approximation function.
            >>> tan(pCube1.t)  # returns a network which passes pCube1's [tx, ty, tz] into a tan approximation functions.
        """
    
        # https://developer.download.nvidia.com/cg/tan.html
        exp = Expression(container='tan1', 
                         debug=self.debug)
        
        code = '''
        $sin = sin(%s)
        $cos = cos(%s)
        if ($cos != 0, $sin/$cos, 0)
        ''' % (tokens[0], tokens[0])

        result = exp(code)
        self.nodes.extend(exp.getNodes())
        return result

        
    
    @parsedcommand
    def tand(self, tokens):
        """ 
        tan(<input>)
        
            Approximates a tan function (in degrees).
        
            Examples
            --------
            >>> tan(pCube1.tx) # returns a network which passes pCube1's translateX into a tan function.
            >>> tan(pCube1.t)  # returns a network which passes pCube1's [tx, ty, tz] into a tan functions.
        """
    
        # https://developer.download.nvidia.com/cg/tan.html
        exp = Expression(container='tand1', 
                         debug=self.debug)
        
        code = '''
        $sin = sind(%s)
        $cos = cosd(%s)
        if ($cos != 0, $sin/$cos, 0)
        ''' % (tokens[0], tokens[0])

        result = exp(code)
        self.nodes.extend(exp.getNodes())
        return result
        
    
    
    
    @parsedcommand
    def atand(self, tokens):
        """ 
        atand(<input>)
        
            Calculates an arc tan function (in degrees).
        
            Examples
            --------
            >>> acosd(pCube1.tx) # returns a network which passes pCube1's translateX into an arc tan function.
            >>> acosd(pCube1.t)  # returns a network which passes pCube1's [tx, ty, tz] into an arc tan functions.
        """
        
        tokens = self._getPlugs(tokens, compound=False)
        if not len(tokens) in [1,3]:
            raise Exception('trigonometric functions ony supports 1 or 3 plugs')
        
        nodes = []
        exp = Expression(container='atand1', debug=self.debug)
                   
        for token in tokens:
            node = exp._createNode('angleBetween', ss=False)
            mc.setAttr('%s.vector1'%node,1,0,0)
            mc.setAttr('%s.vector2'%node,1,0,0)
            
            angle = '%s.axisAngle.angle'%node # TODO: PUT THIS BACK IN EXPRESSION (FX ATTR EXPANSION BUG)
            
            
            code = '''
            
            # see atan under https://www.chadvernon.com/blog/trig-maya/
        
            $node.vector1Y = $token            
            if ($token < 0, -$angle, $angle)
            
            '''
            
            result  = exp(code, variables=locals())[0]
            nodes.append(result)
        
        # make result a vector if there are 3 outputs
        result = nodes[0]
        if len(nodes) == 3:
            vec = exp._double3()
            for i, xyz in enumerate(['X', 'Y', 'Z']):
                mc.connectAttr(nodes[i], '%s%s' % (vec, xyz), f=True)
                
            result = vec
 
        self.nodes.extend(exp.getNodes())           
        return result    
    
    
    
    @parsedcommand
    def atan(self, tokens):
        """ 
        atan(<input>)
        
            Calculates an arc tan function (in degrees).
        
            Examples
            --------
            >>> asin(pCube1.tx) # returns a network which passes pCube1's translateX into an arc tan approximation function.
            >>> asin(pCube1.t)  # returns a network which passes pCube1's [tx, ty, tz] into an arc tan approximation functions.
        """
        
    
        # pack in it's own container to reduce the clutter
        exp = Expression(container='atan1', 
                         debug=self.debug)
        
        result = exp('radians(atand(%s))' % tokens[0])
        self.nodes.extend(exp.getNodes())
        
        return result        
    




    # --------------------------- QUATERNION FUNCTIONS --------------------------- #

    def _quatCommon(self, tokens, quat_node, sequential=False, output_attr='outputQuat'):
        """ 
        Quaternion processor utility used by most quaternion functions.
        """
        
        # make sure given items are lists, tuples or sets
        if not isinstance(tokens, (list, tuple, set)):
            tokens = [tokens]
            
        if not sequential:
            if len(tokens) != 1:
                raise Exception('%s requires 1 input, given: %s' % (quat_node, tokens))
        else:
            if len(tokens) < 2:
                raise Exception('%s requires multiple inputs, given: %s' % (quat_node, tokens))
    
        # Test inputs for quaternions, if matrix given
        # do a conversion for convenience.
        for i, item in enumerate(tokens):
            if self._isMatrixAttr(item):
                tokens[i] = self.matrixToQuat(item)
    
            elif not self._isQuatAttr(item):
                raise Exception('%s requires quaternions, given: %s' % (quat_node, tokens))
    
        node = self._createNode(quat_node, ss=True)
        if sequential:
            self._connectAttr(tokens[0], '%s.input1Quat' % node)
            self._connectAttr(tokens[1], '%s.input2Quat' % node)
    
            for item in tokens[2:]:
                node_ = self._createNode(quat_node, ss=True)
                self._connectAttr('%s.outputQuat' % node, '%s.input1Quat' % node_)
                self._connectAttr(item, '%s.input2Quat' % node_)
                node = node_
    
        else:
            self._connectAttr(tokens[0], '%s.inputQuat' % node)
    
        return '%s.%s' % (node, output_attr)
    
    @parsedcommand
    def quatAdd(self, tokens):
        """ 
        quatAdd(<input>, <input>, <input>, ...)
        
            Returns the sum of added quaternions.
        
            Examples
            --------
            >>> quatAdd(pCube1.rq, pCube1.rq)
        """
        return self._quatCommon(tokens, 'quatAdd', sequential=True)
    
    @parsedcommand
    def quatProd(self, tokens):
        """ 
        quatProd(<input>, <input>, <input>, ...)
        
            Returns the product of multiplied quaternions.
        
            Examples
            --------
            >>> quatProd(pCube1.rq, pCube2.rq)
        """
        return self._quatCommon(tokens, 'quatProd', sequential=True)
    
    @parsedcommand
    def quatSub(self, tokens):
        """ 
        quatSub(<input>, <input>, <input>, ...)
        
            Returns the sum of subtracted quaternions.
        
            Examples
            --------
            >>> quatSub(pCube1.rq, pCube1.rq)
        """
        return self._quatCommon(tokens, 'quatSub', sequential=True)
    
    @parsedcommand
    def quatNegate(self, tokens):
        """ 
        quatNegate(<input>)
        
            Negates a quaternion.
        
            Examples
            --------
            >>> quatNegate(pCube1.wm)
        """
        return self._quatCommon(tokens, 'quatNegate')
    
    @parsedcommand
    def quatToEuler(self, tokens):
        """ 
        quatToEuler(<input>)
        
            Turns a quaternion into a euler angle.
        
            Examples
            --------
            >>> quatToEuler(pCube1.wm)
        """
        return self._quatCommon(tokens, 'quatToEuler', output_attr='outputRotate')
    
    @parsedcommand
    def eulerToQuat(self, tokens):
        """ 
        eulerToQuat(<euler>,<rotateOrder>)
        
            Turns a euler angle into a guaternion.
        
            Examples
            --------
            >>> eulerToQuat(pCube1.r, some_node.ro)
            >>> eulerToQuat(pCube1.r)
        """
        
        if len(tokens) > 2:
            raise Exception('mag requires max 2 inputs, given: %s' % tokens)
    
        node = self._createNode('eulerToQuat', ss=True)
        self._connectAttr(tokens[0], '%s.inputRotate' % node)
        
        if len(tokens) == 2:
            self._connectAttr(tokens[1], '%s.inputRotateOrder' % node)
        else:
            
            # autoconnect rotate order if present
            obj = tokens[0].split('.')[0]
            
            if mc.attributeQuery('rotateOrder', node=obj, exists=True):
                mc.connectAttr('%s.ro' % obj, '%s.inputRotateOrder' % node)        
    
        return '%s.outputQuat' % node
    
    @parsedcommand    
    def quatNormalize(self, tokens):
        """ 
        quatNormalize(<input>)
        
            Normalizes a quaternion.
        
            Examples
            --------
            >>> quatNormalize(pCube1.wm)
        """
        return self._quatCommon(tokens, 'quatNormalize')
    
    @parsedcommand
    def quatInvert(self, tokens):
        """ 
        quatInvert(<input>)
        
            Inverts a quaternion.
        
            Examples
            --------
            >>> quatInvert(pCube1.wm)
        """
        return self._quatCommon(tokens, 'quatInvert')
    
    @parsedcommand
    def quatConjugate(self, tokens):
        """ 
        quatConjugate(<input>)
        
            Conjugates a quaternion.
        
            Examples
            --------
            >>> quatConjugate(pCube1.wm)
        """
        return self._quatCommon(tokens, 'quatConjugate')
    
    @parsedcommand
    def quatSlerp(self, tokens):
        """ 
        quatSlerp(<input>, <input>, ...)
        
            Slerps between two quaternions with optional weight values.
            (default = 0.5)
        
            Examples
            --------
            >>> quatSlerp(pCube1.wm, pCube2.wm)
            >>> quatSlerp(pCube1.wm, pCube2.wm, pCube1.weight)
            
        """
        if len(tokens) <= 1:
            raise Exception('quatSlerp requires 2 or more inputs, given: %s' % tokens)
    
        # parse inputs between matrices and weights
        quats = []
        weights = []
    
        for item in tokens:
    
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

    def _matrixCommon(self, tokens, matrix_node, output_attr='outputMatrix'):
        """ 
        Matrix processor utility.
        """
    
        # make sure given items are lists, tuples or sets
        if isinstance(tokens, ParseResults):
            tokens = tokens.asList()            
            
        elif not isinstance(tokens, (list, tuple, set)):
            tokens = [tokens]
    
        # test input count
        if len(tokens) != 1:
            raise Exception('%s requires 1 input, given: %s' % (matrix_node, tokens))
    
        # make sure items are matrices
        for item in tokens:
            if not self._isMatrixAttr(item):
                raise Exception('%s requires matrices, given: %s' % (matrix_node, tokens))
    
        # process item
        node = self._createNode(matrix_node, ss=True)
        self._connectAttr(tokens[0], '%s.inputMatrix' % node)
    
        return '%s.%s' % (node, output_attr)
    
    
    @parsedcommand
    def matrixDecompose(self, tokens):
        """ 
        matrixDecompose(<input>)
        
            Extracts the position component of a matrix by default.
            Other components can be extracted via variable attribute override.
            Available:
                - .outputTranslate
                - .outputRotate
                - .outputScale
                - .outputShear
                - .inputRotateOrder
            
        
            Examples
            --------
            >>> matrixDecompose(pCube1.wm)
        """
        
        return self._matrixCommon(tokens, 'decomposeMatrix', output_attr='outputTranslate')
    
    
    
    @parsedcommand
    def matrixInverse(self, tokens):
        """ 
        matrixInverse(<input>)
        
            Returns the inverse matrix.
        
            Examples
            --------
            >>> matrixInverse(pCube1.wm)
        """
        return self._matrixCommon(tokens, 'inverseMatrix')
    
    
    #def _matrixInverse(matrix):
        #""" Assumes matrix is 4x4 orthogonal
        #"""
    
        ## Init inverse Matrix
        #m_ = np.empty(matrix.shape)
    
        ## For every matrix
        #for i in range(matrix.shape[0]):
    
            ## Calculate the scale components
            #sx = (matrix[i,0,0]**2 + matrix[i,0,1]**2 + matrix[i,0,2]**2)
            #sy = (matrix[i,1,0]**2 + matrix[i,1,1]**2 + matrix[i,1,2]**2)
            #sz = (matrix[i,2,0]**2 + matrix[i,2,1]**2 + matrix[i,2,2]**2)
    
            ## Normalize scale component
            #m_[i,0,0] = matrix[i,0,0] / sx
            #m_[i,0,1] = matrix[i,1,0] / sx
            #m_[i,0,2] = matrix[i,2,0] / sx
            #m_[i,0,3] = 0.0
            #m_[i,1,0] = matrix[i,0,1] / sy
            #m_[i,1,1] = matrix[i,1,1] / sy
            #m_[i,1,2] = matrix[i,2,1] / sy
            #m_[i,1,3] = 0.0
            #m_[i,2,0] = matrix[i,0,2] / sz
            #m_[i,2,1] = matrix[i,1,2] / sz
            #m_[i,2,2] = matrix[i,2,2] / sz
            #m_[i,2,3] = 0.0
            #m_[i,3,0] = -1 * (m_[i,0,0]*matrix[i,3,0] + m_[i,1,0]*matrix[i,3,1] + m_[i,2,0]*matrix[i,3,2])
            #m_[i,3,1] = -1 * (m_[i,0,1]*matrix[i,3,0] + m_[i,1,1]*matrix[i,3,1] + m_[i,2,1]*matrix[i,3,2])
            #m_[i,3,2] = -1 * (m_[i,0,2]*matrix[i,3,0] + m_[i,1,2]*matrix[i,3,1] + m_[i,2,2]*matrix[i,3,2])
            #m_[i,3,3] = 1.0
    
        #return m_    
    
    
    
    
    @parsedcommand
    def matrixTranspose(self, tokens):
        """ 
        matrixTranspose(<input>)
        
            Returns the transposed matrix.
        
            Examples
            --------
            >>> matrixTranspose(pCube1.wm)
        """
        return self._matrixCommon(tokens, 'transposeMatrix')
    
    @parsedcommand
    def matrixToQuat(self, tokens):
        """ 
        matrixToQuat(<input>)
        
            Converts a matrix into a quaternion.
        
            Examples
            --------
            >>> matrixToQuat(pCube1.wm)
        """
        return self._matrixCommon(tokens, 'decomposeMatrix', output_attr='outputQuat')
    
    
    @parsedcommand
    def matrix(self, tokens):
        """ 
        matrix(<input>, <input>, <input>, <input>)
        
            Constructs a matrix from a list of up to 4 vectors (X,Y,Z,position)
            Where X,Y,Z are the matrix axes
        
            Examples
            --------
            >>> matrix(pCube1.t, pCube2.t, pCube3.t)
            >>> matrix(pCube1.t, pCube2.t, pCube3.t, pCube4.t)
        """
    
        if len(tokens) > 4:
            raise Exception('matrix constructor accepts up to 4 inputs, given: %s' % tokens)
    
        tokens = self._getPlugs(tokens, compound=False)
        
        M = self._createNode('fourByFourMatrix', ss=True)
        for i in range(len(tokens)):
            for j in range(len(tokens[i])):
                if not tokens[i][j]  in [None, 'None']:
                    plug = '%s.in%s%s' % (M, i, j)
                    self._connectAttr(tokens[i][j], plug)
    
        return '%s.output' % M
    

    
    @parsedcommand
    def matrixCompose(self, tokens):
        """ 
        matrixCompose(<translate>, <rotate/quaternion>, <scale>, <shear> <rotateOrder>)
        
            Constructs a matrix from a list of up to 5 inputs.
        
            Examples
            --------
            >>> matrixCompose(pCube1.t, pCube1.r, pCube1.s, None) # pCube1's rotate order will be plugged in
            >>> matrixCompose(pCube1.t, eulerToQuat(pCube1.r), pCube1.s, None) # inputQuaternioon will be used
            >>> matrixCompose(pCube3.t) # identity matrix with just a position
        """
    
        if len(tokens) > 5:
            raise Exception('matrix composer accepts up to 5 inputs, given: %s' % tokens)
    
    
        node = self._createNode('composeMatrix', ss=True)
        plugs0 = ['inputTranslate', 'inputRotate', 'inputScale', 'inputShear', 'inputRotateOrder']
        plugs1 = ['inputTranslate', 'inputQuat',   'inputScale', 'inputShear', 'inputRotateOrder']
        
        for i, item in enumerate(tokens):
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
                        obj = tokens[0].split('.')[0]
                        
                        if mc.attributeQuery('rotateOrder', node=obj, exists=True):
                            mc.connectAttr('%s.ro' % obj, '%s.inputRotateOrder' % node)                                
                    
                else:
                    self._connectAttr(item, '%s.%s'%(node, plugs0[i]))

        return '%s.outputMatrix' % node    
    
    @parsedcommand
    def matrixMult(self, tokens):
        """ 
        matrixMult(<input>, <input>, ...)
        
            Multiplies 2 or more matrices together.
        
            Examples
            --------
            >>> pCube1.wm * pCube2.wm
            >>> matrixMult(pCube1.wm, pCube2.wm, pCube3.wm)
        """
        if len(tokens) <= 1:
            raise Exception('matrixMult requires 2 or more inputs, given: %s' % tokens)
    
        for item in tokens:
            if not self._isMatrixAttr(item):
                raise Exception('matrixMult requires matrices, given: %s' % tokens)
    
        node = self._createNode('multMatrix', ss=True)
    
        for item in tokens:
            self._connectAttr(item, '%s.matrixIn' % node)
    
        return '%s.matrixSum' % node
    
    @parsedcommand
    def matrixAdd(self, tokens):
        """ 
        matrixAdd(<input>, <input>, ...)
        
            Adds matrices together.
        
            Examples
            --------
            >>> pCube1.wm + pCube2.wm
            >>> matrixAdd(pCube1.wm, pCube2.wm, pCube3.wm, ...)
        """
        if len(tokens) <= 1:
            raise Exception('matrixAdd requires 2 or more inputs, given: %s' % tokens)
    
        for item in tokens:
            if not self._isMatrixAttr(item):
                raise Exception('matrixAdd requires matrices, given: %s' % tokens)
    
        node = self._createNode('addMatrix', ss=True)
    
        for item in tokens:
            self._connectAttr(item, '%s.matrixIn' % node)
    
        return '%s.matrixSum' % node
    
    @parsedcommand
    def matrixWeightedAdd(self, tokens):
        """ 
        matrixWeightedAdd(<input>, <input>, ...)
        
            Adds matrices together with optional weight values.
            (default = averaged)
        
            Examples
            --------
            >>> matrixWeightedAdd(pCube1.wm, pCube2.wm, pCube3.wm, ...)
            >>> matrixWeightedAdd(pCube1.wm, pCube2.wm, pCube1.weight, pCube2.weight)
            
        """
        if len(tokens) <= 1:
            raise Exception('matrixAdd requires 2 or more inputs, given: %s' % tokens)
    
        # parse inputs between matrices and weights
        matrices = []
        weights = []
    
        for item in tokens:
    
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
                weights.append(self._double(w, constant=True))
    
        elif matrix_count == 2 and weight_count == 1:
            weights.append(weights[0])
            weights[0] = self.rev([weights[-1]])
    
        elif matrix_count > 1 and weight_count != matrix_count:
            raise Exception('matrixWeightedAdd invalid inputs, given: %s' % tokens)
    
        node = self._createNode('wtAddMatrix', ss=True)
    
        for i in range(matrix_count):
            self._connectAttr(matrices[i], '%s.wtMatrix.matrixIn' % (node))
            self._connectAttr(weights[i], '%s.wtMatrix.weightIn' % (node))
    
        return '%s.matrixSum' % node
    
    
    @parsedcommand
    def vectorMatrixProduct(self, tokens):
        """ 
        vectorMatrixProduct(<input>, <input>)
        
            Creates a vectorProduct node to do a vector matrix product.
        
            Examples
            --------
            >>> pCube1.t * pCube2.wm
            >>> vectorMatrixProduct(pCube1.t, pCube2.wm)
        """
    
        if len(tokens) != 2:
            raise Exception('vectorMatrixProduct requires 2 inputs, given: %s' % tokens)
    
        node = self._createNode('vectorProduct', ss=True)
        mc.setAttr('%s.operation' % node, 3)
        mc.setAttr('%s.normalizeOutput' % node, 0)
    
        matrix0 = self._isMatrixAttr(tokens[0])
        matrix1 = self._isMatrixAttr(tokens[1])
    
        if matrix0 == matrix1:
            raise Exception('vectorMatrixProduct requires a matrix and a vector, given: %s' % tokens)
    
        if matrix0:
            self._connectAttr(tokens[0], '%s.matrix' % node)
        else:
            self._connectAttr(tokens[0], '%s.input1' % node)
    
        if matrix1:
            self._connectAttr(tokens[1], '%s.matrix' % node)
        else:
            self._connectAttr(tokens[1], '%s.input1' % node)
    
        return '%s.output' % node
    
    @parsedcommand
    def vectorMatrixProductNormalized(self, tokens):
        """ 
        vectorMatrixProductNormalized(<input>, <input>)
        
            Creates a normalized vectorProduct node to do a vector matrix product.
        
            Examples
            --------
            >>> vectorMatrixProductNormalized(pCube1.t, pCube2.wm)
        """
    
        if len(tokens) != 2:
            raise Exception('vectorMatrixProductNormalized requires 2 inputs, given: %s' % tokens)
    
        node = self._createNode('vectorProduct', ss=True)
        mc.setAttr('%s.operation' % node, 3)
        mc.setAttr('%s.normalizeOutput' % node, 1)
    
        matrix0 = self._isMatrixAttr(tokens[0])
        matrix1 = self._isMatrixAttr(tokens[1])
    
        if matrix0 == matrix1:
            raise Exception('nVectorMatrixProduct requires a matrix and a vector, given: %s' % tokens)
    
        if matrix0:
            self._connectAttr(tokens[0], '%s.matrix' % node)
        else:
            self._connectAttr(tokens[0], '%s.input1' % node)
    
        if matrix1:
            self._connectAttr(tokens[1], '%s.matrix' % node)
        else:
            self._connectAttr(tokens[1], '%s.input1' % node)
    
        return '%s.output' % node
    
    @parsedcommand
    def pointMatrixProduct(self, tokens):
        """ 
        pointMatrixProduct(<input>, <input>)
        
            Creates a vectorProduct node to do a point matrix product.
        
            Examples
            --------
            >>> pointMatrixProduct(pCube1.t, pCube2.wm)
        """
    
        if len(tokens) != 2:
            raise Exception('pointMatrixProduct requires 2 inputs, given: %s' % tokens)
    
        node = self._createNode('vectorProduct', ss=True)
        mc.setAttr('%s.operation' % node, 4)
        mc.setAttr('%s.normalizeOutput' % node, 0)
    
        matrix0 = self._isMatrixAttr(tokens[0])
        matrix1 = self._isMatrixAttr(tokens[1])
    
        if matrix0 == matrix1:
            raise Exception('pointMatrixProduct requires a matrix and a vector, given: %s' % tokens)
    
        if matrix0:
            self._connectAttr(tokens[0], '%s.matrix' % node)
        else:
            self._connectAttr(tokens[0], '%s.input1' % node)
    
        if matrix1:
            self._connectAttr(tokens[1], '%s.matrix' % node)
        else:
            self._connectAttr(tokens[1], '%s.input1' % node)
    
        return '%s.output' % node
    
    @parsedcommand
    def matrixAdd(self, tokens):
        """ 
        matrixAdd(<input>, <input>, ...)
        
            Adds matrices together.
        
            Examples
            --------
            >>> pCube1.wm + pCube2.wm
            >>> matrixAdd(pCube1.wm, pCube2.wm, pCube3.wm, ...)
        """
        if len(tokens) <= 1:
            raise Exception('matrixAdd requires 2 or more inputs, given: %s' % tokens)
    
        for item in tokens:
            if not self._isMatrixAttr(item):
                raise Exception('matrixAdd requires matrices, given: %s' % tokens)
    
        node = self._createNode('addMatrix', ss=True)
    
        for item in tokens:
            self._connectAttr(item, '%s.matrixIn' % node)
    
        return '%s.matrixSum' % node
    
    @parsedcommand
    def matrixWeightedAdd(self, tokens):
        """ 
        matrixWeightedAdd(<input>, <input>, ...)
        
            Adds matrices together with optional weight values.
            (default = averaged)
        
            Examples
            --------
            >>> matrixWeightedAdd(pCube1.wm, pCube2.wm, pCube3.wm, ...)
            >>> matrixWeightedAdd(pCube1.wm, pCube2.wm, pCube1.weight, pCube2.weight)
            
        """
        if len(tokens) <= 1:
            raise Exception('matrixAdd requires 2 or more inputs, given: %s' % tokens)
    
        # parse inputs between matrices and weights
        matrices = []
        weights = []
    
        for item in tokens:
    
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
                weights.append(self._double(w, constant=True))
    
        elif matrix_count == 2 and weight_count == 1:
            weights.append(weights[0])
            weights[0] = self.rev([weights[-1]])
    
        elif matrix_count > 1 and weight_count != matrix_count:
            raise Exception('matrixWeightedAdd invalid inputs, given: %s' % tokens)
    
        node = self._createNode('wtAddMatrix', ss=True)
    
        for i in range(matrix_count):
            self._connectAttr(matrices[i], '%s.wtMatrix.matrixIn' % (node))
            self._connectAttr(weights[i], '%s.wtMatrix.weightIn' % (node))
    
        return '%s.matrixSum' % node
    
    @parsedcommand
    def matrixMult(self, tokens):
        """ 
        matrixMult(<input>, <input>, ...)
        
            Multiplies 2 or more matrices together.
        
            Examples
            --------
            >>> pCube1.wm * pCube2.wm
            >>> matrixMult(pCube1.wm, pCube2.wm, pCube3.wm)
        """
        if len(tokens) <= 1:
            raise Exception('matrixMult requires 2 or more inputs, given: %s' % tokens)
    
        for item in tokens:
            if not self._isMatrixAttr(item):
                raise Exception('matrixMult requires matrices, given: %s' % tokens)
    
        node = self._createNode('multMatrix', ss=True)
    
        for item in tokens:
            self._connectAttr(item, '%s.matrixIn' % node)
    
        return '%s.matrixSum' % node
    
    
    @parsedcommand
    def pointMatrixProductNormalized(self, tokens):
        """ 
        pointMatrixProductNormalized(<input>, <input>)
        
            Creates a normalized vectorProduct node to do a point matrix product.
        
            Examples
            --------
            >>> pointMatrixProductNormalized(pCube1.t, pCube2.wm)
        """
    
        if len(tokens) != 2:
            raise Exception('pointMatrixProductNormalized requires 2 inputs, given: %s' % tokens)
    
        node = self._createNode('vectorProduct', ss=True)
        mc.setAttr('%s.operation' % node, 4)
        mc.setAttr('%s.normalizeOutput' % node, 1)
    
        matrix0 = self._isMatrixAttr(tokens[0])
        matrix1 = self._isMatrixAttr(tokens[1])
    
        if matrix0 == matrix1:
            raise Exception('pointMatrixProductNormalized requires a matrix and a vector, given: %s' % tokens)
    
        if matrix0:
            self._connectAttr(tokens[0], '%s.matrix' % node)
        else:
            self._connectAttr(tokens[0], '%s.input1' % node)
    
        if matrix1:
            self._connectAttr(tokens[1], '%s.matrix' % node)
        else:
            self._connectAttr(tokens[1], '%s.input1' % node)
    
        return '%s.output' % node


    
#VAL = 'pSphere1.t'
#MIN = 0
#MAX = 10
#RESULT = 'pCube1.t'

#exp = '''
#$delta  = ($VAL - $MIN)
#$range  = ($MAX - $MIN)
#$test   = ($delta/$range)
#$ratio  = 1 - exp(-1 * abs($test))
#$result = $MIN + ($ratio * $range * sign($test))
##$RESULT = if ($result<$MIN, $VAL, $result)
#'''


#e = Expression(consolidate=True, debug=True)
#print e(exp, variables=locals())





#code = '''

#$SINE   = sind(HEART_PULSE.pulse*360) * HEART_PULSE.pulseScale
#$COSINE = cosd(HEART_PULSE.pulse*360) * HEART_PULSE.pulseScale
#$AIM    = vector($COSINE-HEART_PULSE.pulseScale,$SINE,1)
#$MAG    = mag($AIM)
#$Z      = $AIM/$MAG

#$Y      = vector(0,1,0)
#$X      = crossNormalized($Y, $Z)
#$Y      = crossNormalized($Z, $X)

#$M      = matrix($X, $Y, $Z)


#$remap  = vector($M.outputRotateZ, $M.outputRotateX, $M.outputRotateY)
#HEART_PULSE.input = sum($remap, HEART_PULSE.r)

#'''


#e = Expression()
#M = e(code)
#mc.select(M)






