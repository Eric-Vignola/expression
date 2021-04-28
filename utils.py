import maya.cmds as mc
import math, re

from pyparsing import Forward, Word, Combine, Literal, Optional, Group, ParseResults
from pyparsing import nums, alphas, alphanums, oneOf, opAssoc, infixNotation, delimitedList



# Define some known constants
CONSTANTS = {'e':math.e,
             'pi':math.pi,
             'twopi':(math.pi*2),
             'phi':((1 + 5 ** 0.5) / 2)}

CONDITIONS = {'==':0,'!=':1,'>':2,'>=':3,'<':4,'<=':5}




def nextFreePlug(node_att):
    ''' Returns the next valid plug
    '''
    
    try:
        split = node_att.split('.')
        att   = split[-1]
        node  = '.'.join(split[:-1])
        index = 0
        
        while True:
            plugged = mc.listConnections('%s.%s[%s]'%(node,att,index))
            if not plugged:
                break
            index+=1
        
        return '%s.%s[%s]'%(node,att,index)
    
    except:
        return node_att



def listPlugs(node_att, next_free_plug=True):
    #print 'listPlugs: %s'%node_att
    if node_att and mc.objExists(node_att):
        if next_free_plug:
            node_att = nextFreePlug(node_att)
        node     = node_att.split('.')
        node     = '.'.join(node[:-1])    
        return ['%s.%s'%(node,x) for x in mc.listAttr(node_att)]
    
    else:
        return [node_att] # to handle non plug items, like conditions "<>!="
    


def getPlugs(items, compound=True):
    
    if not isinstance(items,(list,tuple)):
        items = [items]
    
    
    attrs  = []
    for i,obj in enumerate(items):
        attrs.append(listPlugs(obj, next_free_plug=bool(i)))

    counts = [len(x) for x in attrs]
    maxi   = max(counts)-1
    
    # If all counts the same
    if [counts[0]]*len(counts) == counts and compound:
        return [[x[0]] for x in attrs]
    
    else:
    
        # Compound mode off
        if maxi == 0 and not compound:
            return attrs
            
        
        result = [[None]*maxi for _ in items]
        for i in range(len(items)):
            for j in range(maxi):
                if counts[i] == 1:
                    result[i][j] = attrs[i][0]
                else:
                    result[i][j] = attrs[i][min(counts[i],j+1)]
    
        return result



def isMatrix(item):
    ''' Check if plug is a matrix or not
    '''
    if mc.getAttr(item, type=True) == 'matrix':
        return True

    return False







def connect(src, dst):

    # Connection logic
    # - one to one    x   --> x
    # - one to many   x   --> x,X,r,R
    #                 x   --> y,Y,g,G
    #                 x   --> z,Z,b,B
    # - comp to comp  xyz --> xyz 
    #                 matrix ---> matrix
    #print 'CONNECTING: %s ---> %s'%(src,dst)
    src,dst = getPlugs([src,dst])
    for i in range(len(src)):
        
        #print 'connecting: %s ---> %s'%(src[i],dst[i])
        mc.connectAttr(src[i], dst[i], f=True)
        






def trigonometry(items, x, y, modulo=None, ss=True):
    
    # Handle single value or vector
    items   = getPlugs(items, compound=False)
    results = []
    for i in range(len(items[0])):
        
        plug = items[0][i]
        if modulo:
            plug = eval(str(plug)+'%'+str(modulo))
        
        node = mc.createNode('remapValue', ss=ss)
        connect(plug, '%s.inputValue'%node)
        
        for i in range(len(x)):
            mc.setAttr('%s.value[%s].value_Position'%(node,i),   x[i])
            mc.setAttr('%s.value[%s].value_FloatValue'%(node,i), y[i])
            mc.setAttr('%s.value[%s].value_Interp'%(node,i),     2)
            
        results.append('%s.outValue'%node)
        

    if len(results) == 1:
        return results[0]
    
    elif len(results) == 3:
        vec = vector()
        for i, xyz in enumerate(['X','Y','Z']):
            mc.connectAttr(results[i],'%s%s'%(vec,xyz))
        return vec
        
    else:
        raise Exception('trigonometric functions ony supports 1 or 3 plugs')




    
    
    
    



def multiplyDivide(op, items, ss=True, lock=True):
    
    mat0 = isMatrix(items[0])
    mat1 = isMatrix(items[1])
    
    if not mat0 and not mat1:
        
        # modulo
        if op == '%':
            #floor = xyz - floor(xyz/div)*div
            return eval('%s - floor(%s/%s)*%s'%(items[0],items[0],items[1],items[1]))
        
        # floor division
        elif op == '//':
            return eval('floor(%s/%s)'%(items[0],items[1]))
        
        
        else:
            node = mc.createNode('multiplyDivide', ss=ss)
        
            # multiply
            if op == '*':
                mc.setAttr('%s.operation'%node,1,lock=lock)
                
            # divide
            elif op == '/':
                mc.setAttr('%s.operation'%node,2,lock=lock)
                
            # power
            elif op == '**':
                mc.setAttr('%s.operation'%node,3,lock=lock)
                
            else:
                raise Exception('unsupported operator: %s'%op)
        
        
        connect(items[0],'%s.input1'%node)
        connect(items[1],'%s.input2'%node)
        
        return '%s.output'%node 
    
    
    else:
        
        # is this a mat * mat
        if mat0 and mat1:
            return _matrixMultiply(items)
        
        # is this a mat * p
        else:
            return _vectorMatrixProduct(items)



def plusMinusAverage(op, items, lock=True):
    node = mc.createNode('plusMinusAverage',ss=True)
    
    # plus
    if op == '+':
        mc.setAttr('%s.operation'%node,1,lock=lock)
        
    # minus
    elif op == '-':
        mc.setAttr('%s.operation'%node,2,lock=lock)
        
    # average
    elif op == 'avg':
        mc.setAttr('%s.operation'%node,3,lock=lock)
        
    else:
        raise Exception('unsupported operator: %s'%op)
        
        
    # Connect
    for i, obj in enumerate(items):
        connect(obj,'%s.input3D[%s]'%(node,i))
 
        
    return '%s.output3D'%node



def constant(value=0, ss=True, at='double', name='constant1'):
    node = mc.createNode('network', name=name, ss=ss)
    mc.addAttr(node, ln='value',  at=at, dv=float(value), keyable=True)

    return '%s.value'%node
        


def vector(xyz=[0,0,0], ss=True, at='double', name='vector1'):
    xyz = [float(x) for x in xyz]
    node = mc.createNode('network', name=name, ss=ss)
    
    mc.addAttr(node,ln='value',  at='%s3'%at, keyable=True)
    mc.addAttr(node,ln='valueX', at=at,  p='value', dv=xyz[0], keyable=True)
    mc.addAttr(node,ln='valueY', at=at,  p='value', dv=xyz[1], keyable=True)
    mc.addAttr(node,ln='valueZ', at=at,  p='value', dv=xyz[2], keyable=True)
    
    return '%s.value'%node





def operatorOperands(tokenlist):
    "generator to extract operators and operands in pairs"
    it = iter(tokenlist)
    while 1:
        try:
            yield (next(it), next(it))
        except StopIteration:
            break


class EvalSignOp(object):
    "Class to evaluate expressions with a leading + or - sign"
    
    def __init__(self, tokens):
        self.sign, self.value = tokens[0]
        
    def eval(self):
        if self.sign == '-':
            return multiplyDivide('*', [constant(-1),self.value.eval()])
        else:
            return self.value.eval()
    
    

class EvalPowerOp(object):
    "Class to evaluate multiplication and division expressions"
    
    def __init__(self, tokens):
        self.value = tokens[0]
        
    def eval(self):
        res = self.value[-1].eval()
        for val in self.value[-3::-2]:
            res = multiplyDivide('**', [val.eval(),res])
        return res
    

class EvalMultOp(object):
    "Class to evaluate multiplication and division expressions"
    
    def __init__(self, tokens):
        self.value = tokens[0]
        
    def eval(self):
        prod = self.value[0].eval()
        for op,val in operatorOperands(self.value[1:]):        
            prod = multiplyDivide(op, [prod,val.eval()])
        
        return prod
    

class EvalAddOp(object):
    "Class to evaluate addition and subtraction expressions"
    
    def __init__(self, tokens):
        self.value = tokens[0]
        
    def eval(self):
        prod = self.value[0].eval()
        for op,val in operatorOperands(self.value[1:]):        
            prod = plusMinusAverage(op, [prod,val.eval()])
        
        return prod


class EvalComparisonOp(object):
    "Class to evaluate comparison expressions"

    def __init__(self, tokens):
        self.value = tokens[0]
        
    def eval(self):
        
        if self.value[1] == '=':
            src = self.value[2].eval()
            dst = self.value[0].eval()
            
            connect(src,dst)
            #return True
        #return False
        return src



class EvalElement(object):
    "Class to evaluate a parsed constant or variable"
    
    def __init__(self, tokens):
        self.value = tokens[0]
        
    def eval(self):
        #print 'EvalElement: %s'%self.value
        
        # Is this a function? eg ['sin', [<__main__.EvalAddOp object,...]]
        if isinstance(self.value, ParseResults):
            
            function = self.value[0]
            plugs    = []
            for item in self.value[1]:
                plugs.append(item.eval())
                
            return FUNCTIONS[function](plugs)
        
        
        # Real, Constant, or node.attr 
        else:
            
            # Real
            try:
                value = float(self.value)
                return constant(value, ss=True)
                
            except:
                # CONSTANT
                if self.value in CONSTANTS:
                    return constant(CONSTANTS[self.value], ss=True)
                
                # node.attr
                elif mc.objExists(self.value):
                    return self.value
                
                # This could be a string variable used by a function.
                # return as is.
                else:
                    return self.value
        

        
# ------------------------------- FUNCTIONS ------------------------------- #


def _easeIn(items):
    ''' Creates a trigonometric function that approximates an easeIn function
    '''    
    return trigonometry(items, x=[0,1], y=[0,1])

def _easeOut(items):
    ''' Creates a trigonometric function that approximates an easeOut function
    '''    
    return trigonometry(items, x=[1,0], y=[0,1])

def _sin(items, pi=math.pi):
    ''' Creates a trigonometric function that approximates a sine function
    '''       
    pi = math.pi
    x  = [(-5*pi/2),(5*pi/2),(-3*pi/2),(-1*pi/2),(pi/2),(3*pi/2)]
    y  = [   -1    ,    1   ,     1   ,    -1   ,   1  ,   -1    ]
    return trigonometry(items, x, y, modulo=2*pi)

def _sind(items):
    ''' Creates a trigonometric function that approximates a sine function (in degrees)
    '''      
    return _sin(items, pi=180.)

def _cos(items, pi=math.pi):
    ''' Creates a trigonometric function that approximates a cosine function
    '''       
    pi = math.pi
    x  = [(-2*pi), (2*pi), (-1*pi), 0, pi]
    y  = [   1   ,    1  ,   -1   , 1, -1]
    return trigonometry(items, x, y, modulo=2*pi)

def _cosd(items):
    ''' Creates a trigonometric function that approximates a cosine function (in degrees)
    '''      
    return _cos(items, pi=180.)


def _acos(items):

    '''
    float acos(float x) {
    float negate = float(x < 0);
    x = abs(x);
    float ret = -0.0187293;
    ret = ret * x;
    ret = ret + 0.0742610;
    ret = ret * x;
    ret = ret - 0.2121144;
    ret = ret * x;
    ret = ret + 1.5707288;
    ret = ret * sqrt(1.0-x);
    ret = ret - 2 * negate * ret;
    return negate * 3.14159265358979 + ret;  
    '''


def _asin(items):
    
    '''
    float negate = float(x < 0);
    x = abs(x);
    float ret = -0.0187293;
    ret *= x;
    ret += 0.0742610;
    ret *= x;
    ret -= 0.2121144;
    ret *= x;
    ret += 1.5707288;
    ret = 3.14159265358979*0.5 - sqrt(1.0 - x)*ret;
    return ret - 2 * negate * ret;
        
    '''
    
    
    
def _tan(items):
    
    '''
    float s, c;
    sincos(a, s, c);
    return s / c;

    '''





def _magnitude(items):
    ''' Creates a distanceBetween node to calculate the magnitude of a vector
    '''
    if len(items) != 1:
        raise Exception('mag requires 1 input, given: %s'%items)
    
    node = mc.createNode('distanceBetween',ss=True)
    connect(items[0],'%s.point1'%node)
    
    return '%s.distance'%node




def _condition(items):
    ''' Creates condition node to sove "if" statements
    '''
    if len(items) != 5:
        raise Exception('cond() needs 5 items: [a,cond_op,b,val if true,val if false]. Given: %s'%items)
    
    condOp = {'==':0,'!=':1,'>':2,'>=':3,'<':4,'<=':5}
    A, op, B, true, false = getPlugs(items)
    

    if len(A)==1 and len(B)==1:
        node = mc.createNode('condition',ss=True)

        connect(A[0],      '%s.firstTerm'%node)
        mc.setAttr(        '%s.operation'%node, condOp[op[0]], lock=True)
        connect(B[0],      '%s.secondTerm'%node)
        connect(true[0],   '%s.colorIfTrue'%node)
        connect(false[0],  '%s.colorIfFalse'%node)

        return '%s.outColor'%node


    elif len(A[0]) > 1:
        
        vec = vector()
        xyz = listPlugs(vec)[1:]
        for i in range(len(A)):
            connect(_condition([A[i],op[i],B[i],true[i],false[i]]),xyz[i])
        
        return vec


def _clamp(items):
    ''' Creates a _cond based network to clamp a value between min and max if input term == min or max.
        Otherwise use a clamp node. 
    '''
    if len(items) != 3:
        raise Exception('clamp() requires 3 inputs, given: %s'%items)
        
    # all inputs differ, use a clamp node
    if items[0] != items[1] and items[0] != items[2]:
        node = mc.createNode('clamp',ss=True)
        mc.setAttr('%s.renderPassMode'%node, 0, lock=True)
        connect(items[1],'%s.min'%node)
        connect(items[2],'%s.max'%node)
        connect(items[0],'%s.input'%node)
        
        return '%s.output'%node
    
    # shared input, use 2 conditions
    # max(min(my_value, max_value), min_value)
    MIN = _condition([items[0],'<',items[2],items[0],items[2]])
    return _condition([MIN,'>',items[1],MIN,items[1]])



def _dot(items):
    """ Creates a vectorProduct node to do a dot product
    """
    if len(items) != 2:
        raise Exception('dot requires 2 inputs, given: %s'%items)
        
    node = mc.createNode('vectorProduct',ss=True)
    mc.setAttr('%s.operation'%node,1,lock=True)
    mc.setAttr('%s.normalizeOutput'%node,0,lock=True)
    
    connect(items[0],'%s.input1'%node)
    connect(items[1],'%s.input2'%node)
    
    return '%s.output'%node


def _nDot(items):
    """ Creates a vectorProduct node to do a normalized dot product
    """
    if len(items) != 2:
        raise Exception('dot requires 2 inputs, given: %s'%items)
        
    node = mc.createNode('vectorProduct',ss=True)
    mc.setAttr('%s.operation'%node,1,lock=True)
    mc.setAttr('%s.normalizeOutput'%node,1,lock=True)
    
    connect(items[0],'%s.input1'%node)
    connect(items[1],'%s.input2'%node)
    
    return '%s.output'%node


def _cross(items):
    """ Creates a vectorProduct node to do a cross product
    """
    if len(items) != 2:
        raise Exception('dot requires 2 inputs, given: %s'%items)
        
    node = mc.createNode('vectorProduct',ss=True)
    mc.setAttr('%s.operation'%node,2,lock=True)
    mc.setAttr('%s.normalizeOutput'%node,0,lock=True)
    
    connect(items[0],'%s.input1'%node)
    connect(items[1],'%s.input2'%node)
    
    return '%s.output'%node


def _nCross(items):
    """ Creates a vectorProduct node to do a normalized cross product
    """
    if len(items) != 2:
        raise Exception('dot requires 2 inputs, given: %s'%items)
        
    node = mc.createNode('vectorProduct',ss=True)
    mc.setAttr('%s.operation'%node,2,lock=True)
    mc.setAttr('%s.normalizeOutput'%node,1,lock=True)
    
    connect(items[0],'%s.input1'%node)
    connect(items[1],'%s.input2'%node)
    
    return '%s.output'%node



def _unit(items):
    ''' Creates a network to yield a unit vector from a vector
    '''
    if len(items) != 1:
        raise Exception('unit() requires 1 input, given: %s'%items)

    mag    = _magnitude(items)
    mult   = multiplyDivide('/',[items[0], mag], lock=False)
    zero   = constant(0)
    two    = constant(2)
    
    node   = mult.split('.')[0]
    test   = _condition([mag,'==',zero, zero, two])
    bypass = _condition([mag,'==',zero, test, mult])    
    
    connect(test,'%s.operation'%node) # silence div by zero error
    return bypass



def _inverse(items):
    ''' Creates a network to do a 0-x mirror operation
    '''
    if len(items) != 1:
        raise Exception('inv() requires 1 input, given: %s'%items)
        
    if isMatrix(items[0]):
        node = mc.createNode('inverseMatrix',ss=True)
        connect(items[0], '%s.inputMatrix'%node)
        return '%s.outputMatrix'%node
    else:
        return plusMinusAverage('-',[constant(0),items[0]])



def _reverse(items):
    ''' Creates a reverse node to do a 1-x operation
    '''
    if len(items) != 1:
        raise Exception('rev() requires 1 input, given: %s'%items)
        
    node = mc.createNode('reverse',ss=True)
    connect(items[0],'%s.input'%node)
    
    return '%s.output'%node


def _average(items):
    ''' Single node operation to average all items in the list
    '''
    if len(items) < 2:
        raise Exception('avg() requires minimum 2 inputs, given: %s'%items)
        
    return plusMinusAverage('avg',items)


def _sum(items):
    ''' Single node operation to add all items in the list
    '''
    if len(items) < 2:
        raise Exception('sum() requires minimum 2 inputs, given: %s'%items)
        
    return plusMinusAverage('+',items)



def _int(items):
    ''' Turns float to int
        Also correct's Maya's int convention int(1.5) ---> 2  !!!! so wrong !!!!
    '''
    
    if len(items) != 1:
        raise Exception('int() requires 1 input, given: %s'%items)
    
    if len(getPlugs(items)) > 1:
        node = constant(name='int1', at='long')
    else:
        node = vector(name='int1', at='long')
        
        
    f     = constant(0.4999999) # corrent Maya's inappropriate int convention
    zero  = constant(0)
    
    true  = plusMinusAverage('-',[items[0],f])
    false = plusMinusAverage('+',[items[0],f])
    
    test  = _condition([items[0],'>',zero,true,false])
    connect(test, node)
    
    return node


def _max(items):
    ''' Returns the highest value in the list of items
    '''
    if len(items) < 2:
        raise Exception('max() requires minimum 2 inputs, given: %s'%items)    
    
    ret = items[0]
    for obj in items[1:]:
        ret = _condition([ret,'>',obj, ret, obj])
        
    return ret


def _min(items):
    ''' Returns the lowest value in the list of items
    '''
    if len(items) < 2:
        raise Exception('min() requires minimum 2 inputs, given: %s'%items)    
    
    ret = items[0]
    for obj in items[1:]:
        ret = _condition([ret,'<',obj, ret, obj])
        
    return ret


def _sign(items):
    ''' Returns -1 for values < 0
    '''    
    if len(items) != 1:
        raise Exception('sign() requires 1 input, given: %s'%items)        

    return _condition([items[0],'<',constant(0),constant(-1),constant(1)])



def _floor(items):
    ''' Returns floor value of input
    '''  
    
    if len(items) != 1:
        raise Exception('floor() requires 1 input, given: %s'%items)   
    
    
    if len(getPlugs(items)) > 1:
        node = constant(name='int1', at='long')
    else:
        node = vector(name='int1', at='long')
        
    f     = constant(0.4999999) # corrent Maya's inappropriate int convention
    floor = plusMinusAverage('-',[items[0],f])
    connect(floor, node)    

    return node



def _ceil(items):
    ''' Returns ceiling value of input
    '''  
    
    if len(items) != 1:
        raise Exception('floor() requires 1 input, given: %s'%items)   
    
    
    if len(getPlugs(items)) > 1:
        node = constant(name='int1', at='long')
    else:
        node = vector(name='int1', at='long')
        
    f     = constant(0.4999999) # corrent Maya's inappropriate int convention
    floor = plusMinusAverage('+',[items[0],f])
    connect(floor, node)    

    return node


def _dist(items):
    ''' Creates a distanceBetween node to find distance between points or matrices
    '''
    
    if len(items) != 2:
        raise Exception('clamp requires 2 inputs, given: %s'%items)
    
    node = mc.createNode('distanceBetween',ss=True)
    
    if isMatrix(items[0]):
        connect(items[0],'%s.inMatrix1'%node)
    else:
        connect(items[0],'%s.point1'%node)
    
    if isMatrix(items[1]):
        connect(items[1],'%s.inMatrix2'%node)
    else:
        connect(items[1],'%s.point2'%node)
        
    return '%s.distance'%node



def _vectorMatrixProduct(items):
    ''' Creates a vectorProduct node to do a vector matrix product
    '''
    if len(items) != 2:
        raise Exception('vectorMatrixProduct requires 2 inputs, given: %s'%items)
        
    node = mc.createNode('vectorProduct',ss=True)
    mc.setAttr('%s.operation'%node,3,lock=True)
    mc.setAttr('%s.normalizeOutput'%node,0,lock=True)
    
    matrix0 = isMatrix(items[0])
    matrix1 = isMatrix(items[1])
    
    if matrix0 == matrix1:
        raise Exception('vectorMatrixProduct requires a matrix and a vector, given: %s'%items)
        
    if matrix0:
        connect(items[0],'%s.matrix'%node)
    else:
        connect(items[0],'%s.input1'%node)
        
    if matrix1:
        connect(items[1],'%s.matrix'%node)
    else:
        connect(items[1],'%s.input1'%node)
    
    return '%s.output'%node



def _nVectorMatrixProduct(items):
    ''' Creates a vectorProduct node to do a normalized vector matrix product
    '''
    if len(items) != 2:
        raise Exception('nVectorMatrixProduct requires 2 inputs, given: %s'%items)
        
    node = mc.createNode('vectorProduct',ss=True)
    mc.setAttr('%s.operation'%node,3,lock=True)
    mc.setAttr('%s.normalizeOutput'%node,1,lock=True)
    
    matrix0 = isMatrix(items[0])
    matrix1 = isMatrix(items[1])
    
    if matrix0 == matrix1:
        raise Exception('nVectorMatrixProduct requires a matrix and a vector, given: %s'%items)
        
    if matrix0:
        connect(items[0],'%s.matrix'%node)
    else:
        connect(items[0],'%s.input1'%node)
        
    if matrix1:
        connect(items[1],'%s.matrix'%node)
    else:
        connect(items[1],'%s.input1'%node)
    
    return '%s.output'%node



def _pointMatrixProduct(items):
    ''' Creates a vectorProduct node to do a point matrix product
    '''
    if len(items) != 2:
        raise Exception('pointMatrixProduct requires 2 inputs, given: %s'%items)
        
    node = mc.createNode('vectorProduct',ss=True)
    mc.setAttr('%s.operation'%node,4,lock=True)
    mc.setAttr('%s.normalizeOutput'%node,0,lock=True)
    
    matrix0 = isMatrix(items[0])
    matrix1 = isMatrix(items[1])
    
    if matrix0 == matrix1:
        raise Exception('pointMatrixProduct requires a matrix and a vector, given: %s'%items)
        
    if matrix0:
        connect(items[0],'%s.matrix'%node)
    else:
        connect(items[0],'%s.input1'%node)
        
    if matrix1:
        connect(items[1],'%s.matrix'%node)
    else:
        connect(items[1],'%s.input1'%node)
    
    return '%s.output'%node





def _matrixMultiply(items):
    ''' Multiplies two matrices together
    '''
    if len(items) != 2:
        raise Exception('vectorMatrixProduct requires 2 inputs, given: %s'%items)
        
    node = mc.createNode('multMatrix',ss=True)

    matrix0 = isMatrix(items[0])
    matrix1 = isMatrix(items[1])
    
    if matrix0 == matrix1 == False:
        raise Exception('matrixMultiply requires matrices, given: %s'%items)
     
        
    connect(items[0],'%s.matrixIn'%node)
    connect(items[1],'%s.matrixIn'%node)

    
    return '%s.matrixSum'%node  


def _nPointMatrixProduct(items):
    ''' Creates a vectorProduct node to do a normalized point matrix product
    '''
    if len(items) != 2:
        raise Exception('nPointMatrixProduct requires 2 inputs, given: %s'%items)
        
    node = mc.createNode('vectorProduct',ss=True)
    mc.setAttr('%s.operation'%node,4,lock=True)
    mc.setAttr('%s.normalizeOutput'%node,1,lock=True)
    
    matrix0 = isMatrix(items[0])
    matrix1 = isMatrix(items[1])
    
    if matrix0 == matrix1:
        raise Exception('nPointMatrixProduct requires a matrix and a vector, given: %s'%items)
        
    if matrix0:
        connect(items[0],'%s.matrix'%node)
    else:
        connect(items[0],'%s.input1'%node)
        
    if matrix1:
        connect(items[1],'%s.matrix'%node)
    else:
        connect(items[1],'%s.input1'%node)
    
    return '%s.output'%node



def _vector(items):
    ''' Creates a vector out of inputs
    '''
    if len(items) != 3:
        raise Exception('vector requires 3 inputs, given: %s'%items)    

    node = vector(name='vector1', at='double')
    connect(items[0],'%sX'%node)
    connect(items[1],'%sY'%node)
    connect(items[2],'%sZ'%node)
    
    return node



def _abs(items):
    items = getPlugs(items, compound=False)[0]
    
    if not len(items) in [1,3]:
        raise Exception('abs works on 1 or 3 inputs, given: %s'%items)    
    
    result = []
    zero = constant(0)
    for i in items:
        neg = eval('-1*%s'%i)
        test = _condition([i,'<',zero,neg,i])
        result.append(test)
    
    if len(result) > 1:
        return _vector(result)
    
    else:
        return result[0]
    
    




FUNCTIONS = {'abs':                  _abs,
             'avg':                  _average,
             'ceil':                 _ceil,
             'clamp':                _clamp,
             'cos':                  _cos,
             'cosd':                 _cosd,
             'cross':                _cross,
             'dist':                 _dist,
             'dot':                  _dot,
             'easeIn':               _easeIn,
             'easeOut':              _easeOut,
             'floor':                _floor,
             'if':                   _condition,
             'int':                  _int,
             'inv':                  _inverse,
             'mag':                  _magnitude,
             'max':                  _max,
             'matrixMultiply':       _matrixMultiply,
             'min':                  _min,
             'pointMatrixProduct':   _pointMatrixProduct,
             'rev':                  _reverse,
             'sign':                 _sign,
             'sin':                  _sin,
             'sind':                 _sind,
             'sum':                  _sum,
             'unit':                 _unit,
             'vector':               _vector,
             'vectorMatrixProduct':  _vectorMatrixProduct,
             
             'nCross':               _nCross,
             'nDot':                 _nDot,
             'nPointMatrixProduct':  _nPointMatrixProduct,
             'nVectorMatrixProduct': _nVectorMatrixProduct}




def evaluate_line(exp):
    
    # define the parser
    expression = Forward()
    var        = Word(alphas)    
    lpar       = Literal( '(' ).suppress()
    rpar       = Literal( ')' ).suppress()
    function   = Group(var + 
                       lpar + 
                       Group(Optional(delimitedList(expression))) + 
                       rpar)
    
    condition  = oneOf('< <= > >= == !=')
    
    integer    = Word(nums)
    variable   = Word(alphas)
    real       = ( Combine(Word(nums) + Optional( '.' + Word(nums) ) +
                   oneOf( 'e' ) + Optional( oneOf( '+ -' )) + Word(nums)) |
                   Combine(Word(nums) + '.' + Word(nums))
                  )    
    node       = Combine(Word(alphanums+'_:')+'.'+Word(alphanums+'_[].'))
    
    operand    = function | condition| real | integer | node | variable
    
    signop     = oneOf('+ -')
    multop     = oneOf('* / % //')
    plusop     = oneOf('+ -')
    expop      = Literal('**')
    
    # use parse actions to attach EvalXXX constructors to sub-expressions
    operand.setParseAction(EvalElement)
    expression << infixNotation(operand,
                               [
                                (signop, 1, opAssoc.RIGHT, EvalSignOp),
                                (expop,  2, opAssoc.LEFT,  EvalPowerOp),
                                (multop, 2, opAssoc.LEFT,  EvalMultOp),
                                (plusop, 2, opAssoc.LEFT,  EvalAddOp),
                               ])
    
    # This might break condition function
    comparisonop = oneOf('=')
    comp_expr = infixNotation(expression,
                              [
                              (comparisonop, 2, opAssoc.LEFT, EvalComparisonOp),
                              ])
                   
                   
    # Patch fix for condition function             
    # cond(a>=b,t,f) --> cond(a,>=,b,t,f)
    regex = r"([^,])(>=|<=|!=|==|>|<)([^,])"
    replace = r"\1,\2,\3"
    exp = re.sub(regex, replace, exp)
        
    ret = comp_expr.parseString(exp)[0]
    
    
    #print 'EVAL: %s'%exp
    return ret.eval()



def eval(expression, variables=None):
    
    if not isinstance(expression, (list, tuple)):
        expression = str(expression).splitlines()
        
    # init known variables    
    known_variables = {}
    if variables:
        known_variables.update(variables)
    
    
    result = None
    for line in expression:

        # ignore lines that are commented out, or trailing comments
        if not line.strip().startswith('#'):
            
            if '#' in line:
                line = line.split('#')[0]
    
            
            # --- Process variables --- #
            # Are we piping a result into a variables (line will begin with $... =)
            stored = re.findall(r'\$.+?=', line)
            if stored:
                stored = stored[0]
                if '.' in stored:
                    stored=None # this is a normal variable bubstitution
                else:
                    line = line.replace(stored, '') # stored var will be set at the end
                    stored = stored[1:-1].strip()
                
                
            # process known variables
            if known_variables:
                v = sorted([x for x in re.findall("[\d$A-Za-z]*", line) if x.startswith('$')])[::-1]
                for var in v:
                    if var[1:] in known_variables:
                        line = line.replace(var, known_variables[var[1:]])
    
    
            # --- evaluate the line --- # 
            line = line.strip()
            if line:
                result = evaluate_line(line)
                
            # store the result
            if stored:
                known_variables[stored] = result
            
            
    return result




# example usage
#eval('pCube1.t = vector(time1.o, sind(time1.o/360 * 90) * 5, 0)')
