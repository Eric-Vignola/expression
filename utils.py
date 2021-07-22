import cmd
import math
import random
import re

import maya.cmds as mc

from pyparsing import Forward, Word, Combine, Literal, Optional, Group, ParseResults
from pyparsing import nums, alphas, alphanums, oneOf, opAssoc, infixNotation, delimitedList

# Define some known constants
CONSTANTS = {'e': math.e,
             'pi': math.pi,
             'twopi': (math.pi * 2),
             'halfpi': (math.pi / 2),
             'phi': ((1 + 5 ** 0.5) / 2),
             'None': None}

CONDITIONS = {'==': 0, '!=': 1, '>': 2, '>=': 3, '<': 4, '<=': 5}


def nextFreePlug(query):
    """ 
    Returns the next valid plug index.
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
            return search % index

        index += 1


def listPlugs(query):
    # no need to go any further if not a node.attr
    if not '.' in query:
        return query

    query = nextFreePlug(query)
    node = query.split('.')[0]

    return ['%s.%s' % (node, x) for x in mc.listAttr(query)]


def getPlugs(query, compound=True):
    """
    Enumerates input plugs.
    ex: compoud=False means that pCube1.t becomes [pCube1.tx, pCube1.ty, pCube1.tz]
    """
    if not isinstance(query, (list, tuple)):
        query = [query]

    attrs = []
    for obj in query:
        attrs.append(listPlugs(obj))

    counts = [len(x) for x in attrs]
    if counts:
        maxi = max(counts) - 1

        # If all counts the same
        # !!! HACK !!! #
        # If one of the inputs is a choice node, we force compound mode
        choice_test = False
        try:
            choice_test = any([mc.nodeType(q) == 'choice' for q in query])
        except:
            pass

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


def isInt(item):
    """ 
    Check if plug is a type int, bool or enum (all valid ints).
    """
    if mc.getAttr(item, type=True) in ['bool', 'int', 'enum']:
        return True

    return False


def isMatrix(item):
    """ 
    Check if plug is a type matrix or not.
    """
    if mc.getAttr(item, type=True) == 'matrix':
        return True

    return False


def isQuaternion(item):
    """ 
    Check if plug is a type quaternion or not.
    """
    if mc.getAttr(item, type=True) in ['double4', 'TdataCompound']:
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
    # print 'CONNECTING: %s ---> %s'%(src,dst)

    src, dst = getPlugs([src, dst])

    for i in range(len(src)):
        # print 'connecting: %s ---> %s'%(src[i],dst[i])
        mc.connectAttr(src[i], dst[i], f=True)


def trigonometry(items, x, y, modulo=None, ss=True):
    """
    Sets up sine/cosine functions using a ramapValue node.
    """
    items = getPlugs(items, compound=False)
    results = []
    for i in range(len(items[0])):

        plug = items[0][i]
        if modulo:
            plug = eval(str(plug) + '%' + str(modulo))

        node = mc.createNode('remapValue', ss=ss)
        connect(plug, '%s.inputValue' % node)

        for j in range(len(x)):
            mc.setAttr('%s.value[%s].value_Position' % (node, j), x[j])
            mc.setAttr('%s.value[%s].value_FloatValue' % (node, j), y[j])
            mc.setAttr('%s.value[%s].value_Interp' % (node, j), 2)

        results.append('%s.outValue' % node)

    if len(results) == 1:
        return results[0]

    elif len(results) == 3:
        vec = vector()
        for i, xyz in enumerate(['X', 'Y', 'Z']):
            mc.connectAttr(results[i], '%s%s' % (vec, xyz), f=True)
        return vec

    else:
        raise Exception('trigonometric functions ony supports 1 or 3 plugs')


def multiplyDivide(op, items, ss=True, lock=True):
    """
    Handles multiply/divide operations on floats, vectors, and matrices.
    """
    mat0 = isMatrix(items[0])
    mat1 = isMatrix(items[1])

    if not mat0 and not mat1:

        # modulo
        if op == '%':
            # floor = xyz - floor(xyz/div)*div
            return eval('%s - floor(%s/%s)*%s' % (items[0], items[0], items[1], items[1]))

        # floor division
        elif op == '//':
            return eval('floor(%s/%s)' % (items[0], items[1]))

        else:
            node = mc.createNode('multiplyDivide', ss=ss)

            # multiply
            if op == '*':
                mc.setAttr('%s.operation' % node, 1, lock=lock)

            # divide
            elif op == '/':
                mc.setAttr('%s.operation' % node, 2, lock=lock)

            # power
            elif op == '**':
                mc.setAttr('%s.operation' % node, 3, lock=lock)

            else:
                raise Exception('unsupported operator: %s' % op)

        connect(items[0], '%s.input1' % node)
        connect(items[1], '%s.input2' % node)

        # Force single output if both inputs are single numerics
        counts = getPlugs(items, compound=False)
        if all(len(x) == 1 for x in counts):
            return '%s.outputX' % node

        return '%s.output' % node

    else:

        # is this a mat * mat
        if mat0 and mat1:
            return _matrixMultiply(items)

        # is this a mat * p
        else:
            return _pointMatrixProduct(items)


def plusMinusAverage(op, items, lock=True):
    """
    Handles plus, minus, average operations on floats and vectors.
    """

    mat0 = isMatrix(items[0])
    mat1 = isMatrix(items[1])

    if not mat0 and not mat1:

        node = mc.createNode('plusMinusAverage', ss=True)

        # plus
        if op == '+':
            mc.setAttr('%s.operation' % node, 1, lock=lock)

        # minus
        elif op == '-':
            mc.setAttr('%s.operation' % node, 2, lock=lock)

        # average
        elif op == 'avg':
            mc.setAttr('%s.operation' % node, 3, lock=lock)

        else:
            raise Exception('unsupported operator: %s' % op)

        # Force single output if both inputs are single numerics
        counts = getPlugs(items, compound=False)
        if all(len(x) == 1 for x in counts):
            for i, obj in enumerate(items):
                connect(obj, '%s.input1D[%s]' % (node, i))

            return '%s.output1D' % node

        # Connect
        for i, obj in enumerate(items):
            connect(obj, '%s.input3D[%s]' % (node, i))

        return '%s.output3D' % node

    else:
        return _matrixSum(items)


def constant(value=0, ss=True, at='double', name='constant1'):
    """
    Creates a numeric value using a network node.
    """
    node = mc.createNode('network', name=name, ss=ss)
    mc.addAttr(node, ln='value', at=at, dv=float(value), keyable=True)

    return '%s.value' % node


def vector(xyz=(0, 0, 0), ss=True, at='double', name='vector1'):
    """
    Defines a vector as a network node with a vector type attribute.
    """
    xyz = [float(x) for x in xyz]
    node = mc.createNode('network', name=name, ss=ss)

    mc.addAttr(node, ln='value', at='%s3' % at, keyable=True)
    mc.addAttr(node, ln='valueX', at=at, p='value', dv=xyz[0], keyable=True)
    mc.addAttr(node, ln='valueY', at=at, p='value', dv=xyz[1], keyable=True)
    mc.addAttr(node, ln='valueZ', at=at, p='value', dv=xyz[2], keyable=True)

    return '%s.value' % node


def operatorOperands(tokenlist):
    """
    PyParsing generator to extract operators and operands in pairs.
    """

    it = iter(tokenlist)
    while 1:
        try:
            yield (next(it), next(it))
        except StopIteration:
            break


class EvalSignOp(object):
    """
    PyParsing class to evaluate expressions with a leading + or - sign.
    """

    def __init__(self, tokens):
        self.sign, self.value = tokens[0]

    def eval(self):
        if self.sign == '-':
            return multiplyDivide('*', [constant(-1), self.value.eval()])
        else:
            return self.value.eval()


class EvalPowerOp(object):
    """
    PyParsing class to evaluate power expressions.
    """

    def __init__(self, tokens):
        self.value = tokens[0]

    def eval(self):
        res = self.value[-1].eval()
        for val in self.value[-3::-2]:
            res = multiplyDivide('**', [val.eval(), res])
        return res


class EvalMultOp(object):
    """
    PyParsing class to evaluate multiplication and division expressions.
    """

    def __init__(self, tokens):
        self.value = tokens[0]

    def eval(self):
        prod = self.value[0].eval()
        for op, val in operatorOperands(self.value[1:]):
            prod = multiplyDivide(op, [prod, val.eval()])

        return prod


class EvalAddOp(object):
    """
    PyParsing class to evaluate addition and subtraction expressions.
    """

    def __init__(self, tokens):
        self.value = tokens[0]

    def eval(self):
        prod = self.value[0].eval()
        for op, val in operatorOperands(self.value[1:]):
            prod = plusMinusAverage(op, [prod, val.eval()])

        return prod


class EvalComparisonOp(object):
    """
    PyParsing class to evaluate comparison expressions.
    """

    def __init__(self, tokens):
        self.value = tokens[0]

    def eval(self):
        if self.value[1] == '=':
            src = self.value[2].eval()
            dst = self.value[0].eval()

            connect(src, dst)

        return src


class EvalElement(object):
    """
    PyParsing class to evaluate a parsed constant or variable.
    """

    def __init__(self, tokens):
        self.value = tokens[0]

    def eval(self):
        # print 'EvalElement: %s'%self.value

        # Is this a function? eg ['sin', [<__main__.EvalAddOp object,...]]
        if isinstance(self.value, ParseResults):

            function = self.value[0]
            plugs = []
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
                    if self.value == 'None':
                        return None
                    return constant(CONSTANTS[self.value], ss=True)

                # node.attr
                elif mc.objExists(self.value):
                    return self.value

                # This could be a string variable used by a function.
                # return as is.
                else:
                    return self.value


# ------------------------------- TRIGONOMETRY ------------------------------- #

def _degrees(items):
    """
    degrees(<input>)
    
        Converts incomming values from radians to degrees.
        (obj in radians * 57.29577951)
    
        Examples
        --------
        >>> degrees(radians(pCube1.rx)) # returns a network which converts rotationX to radians and back to degrees.
        >>> degrees(radians(pCube1.r))  # returns a network which converts [rx, ry, rz] to radians and back to degrees.
    """
    return eval('%s * 57.29577951' % items[0])


def _radians(items):
    """ 
    radians(<input>)
    
        Converts incomming values from degrees to radians.
        (input in degrees * 0.017453292)
    
        Examples
        --------
        >>> radians(pCube1.rx) # returns a network which converts rotationX to radians.
        >>> radians(pCube1.r)  # returns a network which converts [rx, ry, rz] to radians.
    """
    return eval('%s * 0.017453292' % items[0])


# TODO: add built in start, stop remap values
def _easeIn(items):
    """ 
    easeIn(<input>)
    
        Creates an easeIn "tween" function.
    
        Examples
        --------
        >>> easeIn(pCube1.tx) # returns a network which tweens pCube1's translateX value.
        >>> easeIn(pCube1.t)  # returns a network which tweens pCube1's [tx, ty, tz] values.
    """
    return trigonometry(items, x=[0, 1], y=[0, 1])


# TODO: add built in start, stop remap values
def _easeOut(items):
    """ 
    easeOut(<input>)
    
        Creates an easeIn "tween" function.
    
        Examples
        --------
        >>> easeOut(pCube1.tx) # returns a network which tweens pCube1's translateX value.
        >>> easeOut(pCube1.t)  # returns a network which tweens pCube1's [tx, ty, tz] values.
    """
    return trigonometry(items, x=[1, 0], y=[0, 1])


def _sin(items, pi=math.pi):
    """ 
    sin(<input>)
    
        Creates a sine function (in radians).
    
        Examples
        --------
        >>> sin(pCube1.tx) # returns a network which passes pCube1's translateX into a sine function.
        >>> sin(pCube1.t)  # returns a network which passes pCube1's [tx, ty, tz] into a sine functions.
    """
    x = [(-5 * pi / 2), (5 * pi / 2), (-3 * pi / 2), (-1 * pi / 2), (pi / 2), (3 * pi / 2)]
    y = [-1, 1, 1, -1, 1, -1]
    return trigonometry(items, x, y, modulo=2 * pi)


def _sind(items):
    """ 
    sind(<input>)
    
        Creates a sine function (in degrees).
    
        Examples
        --------
        >>> sind(pCube1.tx) # returns a network which passes pCube1's translateX into a sine function.
        >>> sind(pCube1.t)  # returns a network which passes pCube1's [tx, ty, tz] into a sine functions.
    """
    return _sin(items, pi=180.)


def _cos(items, pi=math.pi):
    """ 
    cos(<input>)
    
        Creates a cosine function (in radians).
    
        Examples
        --------
        >>> cos(pCube1.tx) # returns a network which passes pCube1's translateX into a cosine function.
        >>> cos(pCube1.t)  # returns a network which passes pCube1's [tx, ty, tz] into a cosine functions.
    """

    x = [(-2 * pi), (2 * pi), (-1 * pi), 0, pi]
    y = [1, 1, -1, 1, -1]
    return trigonometry(items, x, y, modulo=2 * pi)


def _cosd(items):
    """ 
    cosd(<input>)
    
        Creates a cosine function (in degrees).
    
        Examples
        --------
        >>> cosd(pCube1.tx) # returns a network which passes pCube1's translateX into a cosine function.
        >>> cosd(pCube1.t)  # returns a network which passes pCube1's [tx, ty, tz] into a cosine functions.
    """
    return _cos(items, pi=180.)


def _acos(items):
    """ 
    acos(<input>)
    
        Approximates an arc cosine function (in radians).
    
        Examples
        --------
        >>> acos(pCube1.tx) # returns a network which passes pCube1's translateX into an arc cosine approximation function.
        >>> acos(pCube1.t)  # returns a network which passes pCube1's [tx, ty, tz] into an arc cosine approximation functions.
    """

    # https://developer.download.nvidia.com/cg/acos.html
    items = getPlugs(items, compound=False)
    results = []
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

        results.append(eval(exp))

    if len(results) == 1:
        return results[0]

    elif len(results) == 3:
        vec = vector()
        for i, xyz in enumerate(['X', 'Y', 'Z']):
            # mc.connectAttr(results[i],'%s%s'%(vec,xyz), f=True)
            connect(results[i], '%s%s' % (vec, xyz))
        return vec

    else:
        raise Exception('trigonometric functions ony supports 1 or 3 plugs')


def _asin(items):
    """ 
    asin(<input>)
    
        Approximates an arc sine function (in radians).
    
        Examples
        --------
        >>> asin(pCube1.tx) # returns a network which passes pCube1's translateX into an arc sine approximation function.
        >>> asin(pCube1.t)  # returns a network which passes pCube1's [tx, ty, tz] into an arc sine approximation functions.
    """

    # https://developer.download.nvidia.com/cg/asin.html
    items = getPlugs(items, compound=False)
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

        results.append(eval(exp))

    if len(results) == 1:
        return results[0]

    elif len(results) == 3:
        vec = vector()
        for i, xyz in enumerate(['X', 'Y', 'Z']):
            # mc.connectAttr(results[i],'%s%s'%(vec,xyz), f=True)
            connect(results[i], '%s%s' % (vec, xyz))
        return vec

    else:
        raise Exception('trigonometric functions ony supports 1 or 3 plugs')


def _acosd(items):
    """ 
    acosd(<input>)
    
        Approximates an arc cosine function (in degrees).
    
        Examples
        --------
        >>> acosd(pCube1.tx) # returns a network which passes pCube1's translateX into an arc cosine approximation function.
        >>> acosd(pCube1.t)  # returns a network which passes pCube1's [tx, ty, tz] into an arc cosine approximation functions.
    """
    return eval('degrees(acos(%s))' % items[0])


def _asind(items):
    """ 
    asind(<input>)
    
        Approximates an arc sine function (in radians).
    
        Examples
        --------
        >>> asind(pCube1.tx) # returns a network which passes pCube1's translateX into an arc sine approximation function.
        >>> asind(pCube1.t)  # returns a network which passes pCube1's [tx, ty, tz] into an arc sine approximation functions.
    """
    return eval('degrees(asin(%s))' % items[0])


def _tan(items):
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

    return eval(exp)


def _tand(items):
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

    return eval(exp)


# TODO
def _atan2(items):
    pass
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
def _atan(items):
    pass
    # float atan(float x) {
    #    return _atan2(x, float(1));
    # }


# ------------------------------- UTILITIES ------------------------------- #


# TODO: add input as a time offset
def _frame(items):
    """ 
    frame()
    
        Outputs "current frame" via an infinite linear motion curve.
    
        Examples
        --------
        >>> frame() # returns a current time slider value.
    """

    if items:
        raise Exception('frame functions does not expect inputs')

    curve = mc.createNode('animCurveTU')
    mc.setKeyframe(curve, t=0, v=0.)
    mc.setKeyframe(curve, t=1, v=1.)
    mc.keyTangent(curve, e=True, itt='linear', ott='linear')
    mc.setAttr('%s.preInfinity' % curve, 4)
    mc.setAttr('%s.postInfinity' % curve, 4)

    return '%s.o' % curve


# TODO: this is still experimental.
def _noise(items):
    """ 
    noise(<input>)
    
        Creates a pseudo random function via perlin noise.
    
        Examples
        --------
        >>> noise(pCube1.tx) # Applies noise to pCube1's translateX value.
        >>> noise(pCube1.t)  # Applies noise to pCube1's [tx, ty, tz] values.
    """

    # Handle single value or vector
    items = getPlugs(items, compound=False)
    results = []
    for i in range(len(items[0])):
        plug = items[0][i]

        # create a noise node
        noise = mc.createNode('noise')
        mc.setAttr('%s.ratio' % noise, 1)
        mc.setAttr('%s.noiseType' % noise, 4)  # set noise to 4d perlin (spacetime)
        mc.setAttr('%s.frequencyRatio' % noise, CONSTANTS['phi'])  # set freq to golden value

        # animate the time function with a random seed value
        v0 = random.randint(-1234, 1234) + random.random()
        v1 = v0 + 1
        mc.setKeyframe(noise, attribute='time', t=0, v=v0)
        mc.setKeyframe(noise, attribute='time', t=1, v=v1)
        curve = mc.listConnections('%s.time' % noise)[0]
        mc.keyTangent(curve, e=True, itt='linear', ott='linear')
        mc.setAttr('%s.preInfinity' % curve, 4)
        mc.setAttr('%s.postInfinity' % curve, 4)

        # clamp output because sometimes noise goes beyond 0-1 range
        exp = '%s * clamp(%s.outColorR, 0.001,0.999)' % (plug, noise)
        results.append(eval(exp))

    if len(results) == 1:
        return results[0]

    elif len(results) == 3:
        vec = vector()
        for i, xyz in enumerate(['X', 'Y', 'Z']):
            # mc.connectAttr(results[i],'%s%s'%(vec,xyz), f=True)
            connect(results[i], '%s%s' % (vec, xyz))
        return vec

    else:
        raise Exception('random functions ony supports 1 or 3 plugs')


def _magnitude(items):
    """ 
    mag(<input>)
    
        Returns the magnitude of a vector.
    
        Examples
        --------
        >>> mag(pCube1.t)  # Computes the magnitude of [tx, ty, tz].
    """

    if len(items) != 1:
        raise Exception('mag requires 1 input, given: %s' % items)

    node = mc.createNode('distanceBetween', ss=True)
    connect(items[0], '%s.point1' % node)

    return '%s.distance' % node


def _condition(items):
    """ 
    if(<input> <op> <input>, <input if true>, <input if false>)
    
        Creates condition node to solve "if" statements.
    
        Examples
        --------
        >>> if(pCube1.t > pCube2.t, 0, pCube3.t)
        >>> if(pCube1.rx < 45, pCube1.rx, 45) # outputs pCube1.rx's value with a maximum of 45
    """
    if len(items) != 5:
        raise Exception('cond() needs 5 items: [a,cond_op,b,val if true,val if false]. Given: %s' % items)

    condOp = {'==': 0, '!=': 1, '>': 2, '>=': 3, '<': 4, '<=': 5}
    A, op, B, true, false = getPlugs(items)

    if len(A) == 1 and len(B) == 1:
        node = mc.createNode('condition', ss=True)

        connect(A[0], '%s.firstTerm' % node)
        mc.setAttr('%s.operation' % node, condOp[op[0]], lock=True)
        connect(B[0], '%s.secondTerm' % node)
        connect(true[0], '%s.colorIfTrue' % node)
        connect(false[0], '%s.colorIfFalse' % node)

        return '%s.outColorR' % node


    elif len(A[0]) > 1:

        vec = vector()
        xyz = listPlugs(vec)[1:]
        for i in range(len(A)):
            connect(_condition([A[i], op[i], B[i], true[i], false[i]]), xyz[i])

        return vec


# TODO: Use None to only limit one way, like clamp(pCube1.ty, 0, None)
def _clamp(items):
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
        node = mc.createNode('clamp', ss=True)
        mc.setAttr('%s.renderPassMode' % node, 0, lock=True)
        connect(items[1], '%s.min' % node)
        connect(items[2], '%s.max' % node)
        connect(items[0], '%s.input' % node)

        counts = getPlugs(items, compound=False)
        if all(len(x) == 1 for x in counts):
            return '%s.outputR' % node

        return '%s.output' % node

    # shared input, use 2 conditions
    # max(min(my_value, max_value), min_value)
    MIN = _condition([items[0], '<', items[2], items[0], items[2]])
    return _condition([MIN, '>', items[1], MIN, items[1]])


# TODO: support Matrix dot product?
def _dot(items):
    """ 
    dot(<input>, <input>)
    
        Uses a vectorProduct to do a dot product between two vector inputs.
    
        Examples
        --------
        >>> dot(pCube1.t, pCube2.t)
    """
    if len(items) != 2:
        raise Exception('dot requires 2 inputs, given: %s' % items)

    node = mc.createNode('vectorProduct', ss=True)
    mc.setAttr('%s.operation' % node, 1, lock=True)
    mc.setAttr('%s.normalizeOutput' % node, 0, lock=True)

    connect(items[0], '%s.input1' % node)
    connect(items[1], '%s.input2' % node)

    return '%s.output' % node


def _nDot(items):
    """ 
    ndot(<input>, <input>)
    
        Uses a normalized vectorProduct to do a dot product between two vector inputs.
    
        Examples
        --------
        >>> ndot(pCube1.t, pCube2.t)
    """
    if len(items) != 2:
        raise Exception('dot requires 2 inputs, given: %s' % items)

    node = mc.createNode('vectorProduct', ss=True)
    mc.setAttr('%s.operation' % node, 1, lock=True)
    mc.setAttr('%s.normalizeOutput' % node, 1, lock=True)

    connect(items[0], '%s.input1' % node)
    connect(items[1], '%s.input2' % node)

    return '%s.output' % node


def _cross(items):
    """ 
    cross(<input>, <input>)
    
        Uses a vectorProduct to do a cross product between two vector inputs.
    
        Examples
        --------
        >>> cross(pCube1.t, pCube2.t)
    """
    if len(items) != 2:
        raise Exception('dot requires 2 inputs, given: %s' % items)

    node = mc.createNode('vectorProduct', ss=True)
    mc.setAttr('%s.operation' % node, 2, lock=True)
    mc.setAttr('%s.normalizeOutput' % node, 0, lock=True)

    connect(items[0], '%s.input1' % node)
    connect(items[1], '%s.input2' % node)

    return '%s.output' % node


def _nCross(items):
    """ 
    ncross(<input>, <input>)
    
        Uses a normalized vectorProduct to do a cross product between two vector inputs.
    
        Examples
        --------
        >>> ncross(pCube1.t, pCube2.t)
    """

    if len(items) != 2:
        raise Exception('dot requires 2 inputs, given: %s' % items)

    node = mc.createNode('vectorProduct', ss=True)
    mc.setAttr('%s.operation' % node, 2, lock=True)
    mc.setAttr('%s.normalizeOutput' % node, 1, lock=True)

    connect(items[0], '%s.input1' % node)
    connect(items[1], '%s.input2' % node)

    return '%s.output' % node


def _unit(items):
    """ 
    unit(<input>)
    
        Creates a network that yields a unit vector.
    
        Examples
        --------
        >>> unit(pCube1.t)
    """

    if len(items) != 1:
        raise Exception('unit() requires 1 input, given: %s' % items)

    mag = _magnitude(items)
    mult = multiplyDivide('/', [items[0], mag], lock=False)
    zero = constant(0)
    two = constant(2)

    node = mult.split('.')[0]
    test = _condition([mag, '==', zero, zero, two])
    bypass = _condition([mag, '==', zero, test, mult])

    connect(test, '%s.operation' % node)  # silence div by zero error
    return bypass


def _inverse(items):
    """ 
    inv(<input>)
    
        Creates a network to yields a (0.0-x) mirror operation.
    
        Examples
        --------
        >>> inv(pCube1.t)
    """
    if len(items) != 1:
        raise Exception('inv() requires 1 input, given: %s' % items)

    if isMatrix(items[0]):
        node = mc.createNode('inverseMatrix', ss=True)
        connect(items[0], '%s.inputMatrix' % node)
        return '%s.outputMatrix' % node
    else:
        return plusMinusAverage('-', [constant(0), items[0]])


def _reverse(items):
    """ 
    rev(<input>)
    
        Creates a reverse node to do a (1.0-x) operation.
    
        Examples
        --------
        >>> rev(pCube1.t)
    """
    if len(items) != 1:
        raise Exception('rev() requires 1 input, given: %s' % items)

    node = mc.createNode('reverse', ss=True)
    connect(items[0], '%s.input' % node)

    return '%s.output' % node


def _average(items):
    """ 
    avg(<input>, <input>, <input>, ...)
    
        Single node operation to average all items in the list.
    
        Examples
        --------
        >>> avg(pCube1.t, pCube2.t, pCube3.t, pCube4.t)
    """

    if len(items) < 2:
        raise Exception('avg() requires minimum 2 inputs, given: %s' % items)

    return plusMinusAverage('avg', items)


def _sum(items):
    """ 
    sum(<input>, <input>, <input>, ...)
    
        Single node operation to sum all items in the list.
    
        Examples
        --------
        >>> sum(pCube1.t, pCube2.t, pCube3.t, pCube4.t)
    """

    if len(items) < 2:
        raise Exception('sum() requires minimum 2 inputs, given: %s' % items)

    return plusMinusAverage('+', items)


def _int(items):
    """ 
    int(<input>)
    
        Turns a float value(s) into an int.
    
        Examples
        --------
        >>> int(pCube1.t)
        >>> int(pCube1.tx)
    """

    if len(items) != 1:
        raise Exception('int() requires 1 input, given: %s' % items)

    if len(getPlugs(items[0])[0]) > 1:
        node = vector(name='int1', at='long')
    else:
        node = constant(name='int1', at='long')

    f = constant(0.4999999)  # corrent Maya's inappropriate int convention
    zero = constant(0)

    true = plusMinusAverage('-', [items[0], f])
    false = plusMinusAverage('+', [items[0], f])

    test = _condition([items[0], '>', zero, true, false])
    connect(test, node)

    return node


def _max(items):
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
        ret = _condition([ret, '>', obj, ret, obj])

    return ret


def _min(items):
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
        ret = _condition([ret, '<', obj, ret, obj])

    return ret


def _sign(items):
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

    return _condition([items[0], '<', constant(0), constant(-1), constant(1)])


def _floor(items):
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

    if len(getPlugs(items)) > 1:
        node = constant(name='int1', at='long')
    else:
        node = vector(name='int1', at='long')

    f = constant(0.4999999)  # corrent Maya's inappropriate int convention
    floor = plusMinusAverage('-', [items[0], f])
    connect(floor, node)

    return node


def _ceil(items):
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

    if len(getPlugs(items)) > 1:
        node = constant(name='int1', at='long')
    else:
        node = vector(name='int1', at='long')

    f = constant(0.4999999)  # corrent Maya's inappropriate int convention
    floor = plusMinusAverage('+', [items[0], f])
    connect(floor, node)

    return node


def _dist(items):
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

    node = mc.createNode('distanceBetween', ss=True)

    if isMatrix(items[0]):
        connect(items[0], '%s.inMatrix1' % node)
    else:
        connect(items[0], '%s.point1' % node)

    if isMatrix(items[1]):
        connect(items[1], '%s.inMatrix2' % node)
    else:
        connect(items[1], '%s.point2' % node)

    return '%s.distance' % node


def _abs(items):
    """ 
    abs(<input>)
    
        Outputs the absolute value of a float or vector.
    
        Examples
        --------
        >>> abs(pCube1.t)
        >>> abs(pCube1.tx)
    """

    items = getPlugs(items, compound=False)[0]

    if not len(items) in [1, 3]:
        raise Exception('abs works on 1 or 3 inputs, given: %s' % items)

    result = []
    zero = constant(0)
    for i in items:
        neg = eval('-1*%s' % i)
        test = _condition([i, '<', zero, neg, i])
        result.append(test)

    if len(result) > 1:
        return _vector(result)

    else:
        return result[0]


def _choice(items):
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
    node = mc.createNode('choice', ss=True)

    # plug selector
    if not items[0] is None:
        connect(items[0], '%s.selector' % node)

    # plug inputs
    for item in items[1:]:
        connect(item, '%s.input' % node)

    return '%s.output' % node


def _vector(items):
    """ 
    vector(<input>, <input>, <input>)
    
        Creates a vector out of inputs.
    
        Examples
        --------
        >>> vector(pCube1.tx, pCube2.ty, pCube3.tz)
    """

    if len(items) != 3:
        raise Exception('vector requires 3 inputs, given: %s' % items)

    node = vector(name='vector1', at='double')
    for i, xyz in enumerate(['%s%s' % (node, x) for x in 'XYZ']):

        # skip 'None'
        if not items[i] is None:
            connect(items[i], xyz)

    # connect(items[0], '%sX' % node)
    # connect(items[1], '%sY' % node)
    # connect(items[2], '%sZ' % node)

    return node


# ------------------------------- QUATERNIONS ------------------------------- #

def _quaternion_processor(items, quat_node, sequential=False, output_attr='outputQuat'):
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
        if isMatrix(item):
            items[i] = _matrixToQuaternion([item])

        elif not isQuaternion(item):
            raise Exception('%s requires quaternions, given: %s' % (quat_node, items))

    node = mc.createNode(quat_node, ss=True)
    if sequential:
        connect(items[0], '%s.input1Quat' % node)
        connect(items[1], '%s.input2Quat' % node)

        for item in items[2:]:
            node_ = mc.createNode(quat_node, ss=True)
            connect('%s.outputQuat' % node, '%s.input1Quat' % node_)
            connect(item, '%s.input2Quat' % node_)
            node = node_

    else:
        connect(items[0], '%s.inputQuat' % node)

    return '%s.%s' % (node, output_attr)


def _quaternionSum(items):
    """ 
    quaternionSum(<input>, <input>, <input>, ...)
    
        Returns the sum of added quaternions.
    
        Examples
        --------
        >>> quaternionSum(pCube1.rq, pCube1.rq)
    """
    return _quaternion_processor(items, 'quatAdd', sequential=True)


def _quaternionMultiply(items):
    """ 
    quaternionMultiply(<input>, <input>, <input>, ...)
    
        Returns the product of multiplied quaternions.
    
        Examples
        --------
        >>> quaternionMultiply(pCube1.rq, pCube2.rq)
    """
    return _quaternion_processor(items, 'quatProd', sequential=True)


def _quaternionSub(items):
    """ 
    quaternionSub(<input>, <input>, <input>, ...)
    
        Returns the sum of subtracted quaternions.
    
        Examples
        --------
        >>> quaternionSub(pCube1.rq, pCube1.rq)
    """
    return _quaternion_processor(items, 'quatSub', sequential=True)


def _quaternionNegate(items):
    """ 
    quaternionNegate(<input>)
    
        Negates a quaternion.
    
        Examples
        --------
        >>> quaternionNegate(pCube1.wm)
    """
    return _quaternion_processor(items, 'quatNegate')


def _quaternionToEuler(items):
    """ 
    quaternionToEuler(<input>)
    
        Turns a quaternion into a euler angle.
    
        Examples
        --------
        >>> quaternionToEuler(pCube1.wm)
    """
    return _quaternion_processor(items, 'quatToEuler', output_attr='outputRotate')


def _eulerToQuaternion(items):
    """ 
    eulerToQuaternion(<input>)
    
        Turns a euler angle into a guaternion.
    
        Examples
        --------
        >>> eulerToQuaternion(pCube1.r)
    """
    result = _quaternion_processor(items, 'eulerToQuat')
    if mc.attributeQuery('rotateOrder', items[0], exists=True):
        mc.connectAttr('%s.ro' % items[0], '%s.inputRotateOrder' % result.split('.')[0])


def _quaternionNormalize(items):
    """ 
    quaternionNormalize(<input>)
    
        Normalizes a quaternion.
    
        Examples
        --------
        >>> quaternionNormalize(pCube1.wm)
    """
    return _quaternion_processor(items, 'quatNormalize')


def _quaternionInvert(items):
    """ 
    quaternionInvert(<input>)
    
        Inverts a quaternion.
    
        Examples
        --------
        >>> quaternionInvert(pCube1.wm)
    """
    return _quaternion_processor(items, 'quatInvert')


def _quaternionConjugate(items):
    """ 
    quaternionConjugate(<input>)
    
        Conjugates a quaternion.
    
        Examples
        --------
        >>> quaternionConjugate(pCube1.wm)
    """
    return _quaternion_processor(items, 'quatConjugate')


def _quaternionSlerp(items):
    """ 
    quaternionSlerp(<input>, <input>, ...)
    
        Slerps between two quaternions with optional weight values.
        (default = 0.5)
    
        Examples
        --------
        >>> quaternionSlerp(pCube1.wm, pCube2.wm)
        >>> quaternionSlerp(pCube1.wm, pCube2.wm, pCube1.weight)
        
    """
    if len(items) <= 1:
        raise Exception('quaternionSlerp requires 2 or more inputs, given: %s' % items)

    # parse inputs between matrices and weights
    quats = []
    weights = []

    for item in items:

        # is this a matrix?
        if isMatrix(item):
            quats.append(_matrixToQuaternion([item]))

        elif isQuaternion(item):
            quats.append(item)

        # assume this is a weight
        else:
            weights.append(item)

    node = mc.createNode('quatSlerp', ss=True)

    connect(quats[0], '%s.input1Quat' % (node))
    connect(quats[1], '%s.input2Quat' % (node))

    # if no weights provided, set T to 0.5
    if not weights:
        # weights.append(constant(0.5))
        # connect(weights[0], '%s.inputT'% (node))
        mc.setAttr('%s.inputT' % (node), 0.5)
        
    else:
        connect(weights[0], '%s.inputT'% (node))

    return '%s.outputQuat' % node


# ------------------------------- MATRICES ------------------------------- #

def _matrix_processor(items, matrix_node, output_attr='outputMatrix'):
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
        if not isMatrix(item):
            raise Exception('%s requires matrices, given: %s' % (matrix_node, items))

    # process item
    node = mc.createNode(matrix_node, ss=True)
    connect(items[0], '%s.inputMatrix' % node)

    return '%s.%s' % (node, output_attr)


def _inverseMatrix(items):
    """ 
    inverseMatrix(<input>)
    
        Returns the inverse matrix.
    
        Examples
        --------
        >>> inverseMatrix(pCube1.wm)
    """
    return _matrix_processor(items, 'inverseMatrix')


def _transposeMatrix(items):
    """ 
    transposeMatrix(<input>)
    
        Returns the transposed matrix.
    
        Examples
        --------
        >>> transposeMatrix(pCube1.wm)
    """
    return _matrix_processor(items, 'transposeMatrix')


def _matrixToQuaternion(items):
    """ 
    matrixToQuaternion(<input>)
    
        Converts a matrix into a quaternion.
    
        Examples
        --------
        >>> matrixToQuaternion(pCube1.wm)
    """
    return _matrix_processor(items, 'decomposeMatrix', output_attr='outputQuat')


def _matrix(items):
    """ 
    matrix(<input>, <input>, <input>, <imput>)
    
        Constructs a matrix from a list of up to 4 vectors.
    
        Examples
        --------
        >>> matrix(pCube1.t, pCube2.t, pCube3.t)
        >>> matrix(pCube1.t, pCube2.t, pCube3.t, pCube4.t)
    """

    if len(items) > 4:
        raise Exception('matrix constructor accepts up to 4 inputs, given: %s' % items)

    items = getPlugs(items, compound=False)

    M = mc.createNode('fourByFourMatrix')
    for i in range(len(items)):
        for j in range(len(items[i])):
            if not items[i][j] is None:
                plug = '%s.in%s%s' % (M, i, j)
                connect(items[i][j], plug)

    return '%s.output' % M


def _matrixMultiply(items):
    """ 
    matrixMultiply(<input>, <input>, ...)
    
        Multiplies 2 or more matrices together.
    
        Examples
        --------
        >>> pCube1.wm * pCube2.wm
        >>> matrixMultiply(pCube1.wm, pCube2.wm, pCube3.wm)
    """
    if len(items) <= 1:
        raise Exception('matrixMultiply requires 2 or more inputs, given: %s' % items)

    for item in items:
        if not isMatrix(item):
            raise Exception('matrixMultiply requires matrices, given: %s' % items)

    node = mc.createNode('multMatrix', ss=True)

    for item in items:
        connect(item, '%s.matrixIn' % node)

    return '%s.matrixSum' % node


def _matrixSum(items):
    """ 
    matrixSum(<input>, <input>, ...)
    
        Adds matrices together.
    
        Examples
        --------
        >>> pCube1.wm + pCube2.wm
        >>> matrixSum(pCube1.wm, pCube2.wm, pCube3.wm, ...)
    """
    if len(items) <= 1:
        raise Exception('matrixSum requires 2 or more inputs, given: %s' % items)

    for item in items:
        if not isMatrix(item):
            raise Exception('matrixSum requires matrices, given: %s' % items)

    node = mc.createNode('addMatrix', ss=True)

    for item in items:
        connect(item, '%s.matrixIn' % node)

    return '%s.matrixSum' % node


def _matrixWeightedSum(items):
    """ 
    matrixWeightedSum(<input>, <input>, ...)
    
        Adds matrices together with optional weight values.
        (default = averaged)
    
        Examples
        --------
        >>> matrixWeightedSum(pCube1.wm, pCube2.wm, pCube3.wm, ...)
        >>> matrixWeightedSum(pCube1.wm, pCube2.wm, pCube1.weight, pCube2.weight)
        
    """
    if len(items) <= 1:
        raise Exception('matrixSum requires 2 or more inputs, given: %s' % items)

    # parse inputs between matrices and weights
    matrices = []
    weights = []

    for item in items:

        # is this a matrix?
        if isMatrix(item):
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
            weights.append(constant(w))

    elif matrix_count == 2 and weight_count == 1:
        weights.append(weights[0])
        weights[0] = _reverse([weights[-1]])

    elif matrix_count > 1 and weight_count != matrix_count:
        raise Exception('matrixWeightedSum invalid inputs, given: %s' % items)

    node = mc.createNode('wtAddMatrix', ss=True)

    for i in range(matrix_count):
        connect(matrices[i], '%s.wtMatrix.matrixIn' % (node))
        connect(weights[i], '%s.wtMatrix.weightIn' % (node))

    return '%s.matrixSum' % node



def _vectorMatrixProduct(items):
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

    node = mc.createNode('vectorProduct', ss=True)
    mc.setAttr('%s.operation' % node, 3, lock=True)
    mc.setAttr('%s.normalizeOutput' % node, 0, lock=True)

    matrix0 = isMatrix(items[0])
    matrix1 = isMatrix(items[1])

    if matrix0 == matrix1:
        raise Exception('vectorMatrixProduct requires a matrix and a vector, given: %s' % items)

    if matrix0:
        connect(items[0], '%s.matrix' % node)
    else:
        connect(items[0], '%s.input1' % node)

    if matrix1:
        connect(items[1], '%s.matrix' % node)
    else:
        connect(items[1], '%s.input1' % node)

    return '%s.output' % node


def _nVectorMatrixProduct(items):
    """ 
    nVectorMatrixProduct(<input>, <input>)
    
        Creates a normalized vectorProduct node to do a vector matrix product.
    
        Examples
        --------
        >>> nVectorMatrixProduct(pCube1.t, pCube2.wm)
    """

    if len(items) != 2:
        raise Exception('nVectorMatrixProduct requires 2 inputs, given: %s' % items)

    node = mc.createNode('vectorProduct', ss=True)
    mc.setAttr('%s.operation' % node, 3, lock=True)
    mc.setAttr('%s.normalizeOutput' % node, 1, lock=True)

    matrix0 = isMatrix(items[0])
    matrix1 = isMatrix(items[1])

    if matrix0 == matrix1:
        raise Exception('nVectorMatrixProduct requires a matrix and a vector, given: %s' % items)

    if matrix0:
        connect(items[0], '%s.matrix' % node)
    else:
        connect(items[0], '%s.input1' % node)

    if matrix1:
        connect(items[1], '%s.matrix' % node)
    else:
        connect(items[1], '%s.input1' % node)

    return '%s.output' % node


def _pointMatrixProduct(items):
    """ 
    pointMatrixProduct(<input>, <input>)
    
        Creates a vectorProduct node to do a point matrix product.
    
        Examples
        --------
        >>> pointMatrixProduct(pCube1.t, pCube2.wm)
    """

    if len(items) != 2:
        raise Exception('pointMatrixProduct requires 2 inputs, given: %s' % items)

    node = mc.createNode('vectorProduct', ss=True)
    mc.setAttr('%s.operation' % node, 4, lock=True)
    mc.setAttr('%s.normalizeOutput' % node, 0, lock=True)

    matrix0 = isMatrix(items[0])
    matrix1 = isMatrix(items[1])

    if matrix0 == matrix1:
        raise Exception('pointMatrixProduct requires a matrix and a vector, given: %s' % items)

    if matrix0:
        connect(items[0], '%s.matrix' % node)
    else:
        connect(items[0], '%s.input1' % node)

    if matrix1:
        connect(items[1], '%s.matrix' % node)
    else:
        connect(items[1], '%s.input1' % node)

    return '%s.output' % node


def _matrixSum(items):
    """ 
    matrixSum(<input>, <input>, ...)
    
        Adds matrices together.
    
        Examples
        --------
        >>> pCube1.wm + pCube2.wm
        >>> matrixSum(pCube1.wm, pCube2.wm, pCube3.wm, ...)
    """
    if len(items) <= 1:
        raise Exception('matrixSum requires 2 or more inputs, given: %s' % items)

    for item in items:
        if not isMatrix(item):
            raise Exception('matrixSum requires matrices, given: %s' % items)

    node = mc.createNode('addMatrix', ss=True)

    for item in items:
        connect(item, '%s.matrixIn' % node)

    return '%s.matrixSum' % node


def _matrixWeightedSum(items):
    """ 
    matrixWeightedSum(<input>, <input>, ...)
    
        Adds matrices together with optional weight values.
        (default = averaged)
    
        Examples
        --------
        >>> matrixWeightedSum(pCube1.wm, pCube2.wm, pCube3.wm, ...)
        >>> matrixWeightedSum(pCube1.wm, pCube2.wm, pCube1.weight, pCube2.weight)
        
    """
    if len(items) <= 1:
        raise Exception('matrixSum requires 2 or more inputs, given: %s' % items)

    # parse inputs between matrices and weights
    matrices = []
    weights = []

    for item in items:

        # is this a matrix?
        if isMatrix(item):
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
            weights.append(constant(w))

    elif matrix_count == 2 and weight_count == 1:
        weights.append(weights[0])
        weights[0] = _reverse([weights[-1]])

    elif matrix_count > 1 and weight_count != matrix_count:
        raise Exception('matrixWeightedSum invalid inputs, given: %s' % items)

    node = mc.createNode('wtAddMatrix', ss=True)

    for i in range(matrix_count):
        connect(matrices[i], '%s.wtMatrix.matrixIn' % (node))
        connect(weights[i], '%s.wtMatrix.weightIn' % (node))

    return '%s.matrixSum' % node


def _matrixMultiply(items):
    """ 
    matrixMultiply(<input>, <input>, ...)
    
        Multiplies 2 or more matrices together.
    
        Examples
        --------
        >>> pCube1.wm * pCube2.wm
        >>> matrixMultiply(pCube1.wm, pCube2.wm, pCube3.wm)
    """
    if len(items) <= 1:
        raise Exception('matrixMultiply requires 2 or more inputs, given: %s' % items)

    for item in items:
        if not isMatrix(item):
            raise Exception('matrixMultiply requires matrices, given: %s' % items)

    node = mc.createNode('multMatrix', ss=True)

    for item in items:
        connect(item, '%s.matrixIn' % node)

    return '%s.matrixSum' % node


def _nPointMatrixProduct(items):
    """ 
    nPointMatrixProduct(<input>, <input>)
    
        Creates a normalized vectorProduct node to do a point matrix product.
    
        Examples
        --------
        >>> nPointMatrixProduct(pCube1.t, pCube2.wm)
    """

    if len(items) != 2:
        raise Exception('nPointMatrixProduct requires 2 inputs, given: %s' % items)

    node = mc.createNode('vectorProduct', ss=True)
    mc.setAttr('%s.operation' % node, 4, lock=True)
    mc.setAttr('%s.normalizeOutput' % node, 1, lock=True)

    matrix0 = isMatrix(items[0])
    matrix1 = isMatrix(items[1])

    if matrix0 == matrix1:
        raise Exception('nPointMatrixProduct requires a matrix and a vector, given: %s' % items)

    if matrix0:
        connect(items[0], '%s.matrix' % node)
    else:
        connect(items[0], '%s.input1' % node)

    if matrix1:
        connect(items[1], '%s.matrix' % node)
    else:
        connect(items[1], '%s.input1' % node)

    return '%s.output' % node


def _matrix(items):
    """ 
    matrix(<input>, <input>, <input>, <imput>)
    
        Constructs a matrix from a list of up to 4 vectors.
    
        Examples
        --------
        >>> matrix(pCube1.t, pCube2.t, pCube3.t)
        >>> matrix(pCube1.t, pCube2.t, pCube3.t, pCube4.t)
    """

    if len(items) > 4:
        raise Exception('matrix constructor accepts up to 4 inputs, given: %s' % items)

    items = getPlugs(items, compound=False)

    M = mc.createNode('fourByFourMatrix')
    for i in range(len(items)):
        for j in range(len(items[i])):
            if not items[i][j] is None:
                plug = '%s.in%s%s' % (M, i, j)
                connect(items[i][j], plug)

    return '%s.output' % M


# FUNCTIONS = {}
# for x in dir(trigonometry):
# if not x.startswith('_'):
# FUNCTIONS[x] = getattr(trigonometry, x)

# from types import FunctionType

# FUNCTIONS = {}
# for x in locals().copy():
# if isinstance(locals()[x], FunctionType):
# if not x.startswith('_'):
# FUNCTIONS[x] = locals()[x]


FUNCTIONS = {'abs': _abs,
             'acos': _acos,
             'acosd': _acosd,
             'asin': _asin,
             'asind': _asind,
             'avg': _average,
             'ceil': _ceil,
             'clamp': _clamp,
             'choice': _choice,
             'cos': _cos,
             'cosd': _cosd,
             'cross': _cross,
             'degrees': _degrees,
             'dist': _dist,
             'dot': _dot,
             'easeIn': _easeIn,
             'easeOut': _easeOut,
             'floor': _floor,
             'frame': _frame,
             'if': _condition,
             'int': _int,
             'inv': _inverse,
             'mag': _magnitude,
             'max': _max,
             'matrix': _matrix,
             'matrixSum': _matrixSum,
             'matrixWeightedSum': _matrixWeightedSum,
             'matrixMultiply': _matrixMultiply,
             'matrixToQuaternion': _matrixToQuaternion,
             'min': _min,
             'noise': _noise,
             'pointMatrixProduct': _pointMatrixProduct,
             'quaternionToEuler': _quaternionToEuler,
             'quaternionSum': _quaternionSum,
             'quaternionSub': _quaternionSub,
             'quaternionSlerp': _quaternionSlerp,
             'quaternionNormalize': _quaternionNormalize,
             'quaternionNegate': _quaternionNegate,
             'quaternionMultiply': _quaternionMultiply,
             'quaternionInvert': _quaternionInvert,
             'quaternionConjugate': _quaternionConjugate,
             'radians': _radians,
             'rev': _reverse,
             'sign': _sign,
             'sin': _sin,
             'sind': _sind,
             'sum': _sum,
             'tan': _tan,
             'tand': _tand,
             'unit': _unit,
             'vector': _vector,
             'vectorMatrixProduct': _vectorMatrixProduct,

             'nCross': _nCross,
             'nDot': _nDot,
             'nPointMatrixProduct': _nPointMatrixProduct,
             'nVectorMatrixProduct': _nVectorMatrixProduct}



def usage(query=None, verbose=False):
    """
    Print usage.
    """
    if query in FUNCTIONS:
        print(FUNCTIONS[query].__doc__)

    else:

        if verbose:
            for f in sorted(FUNCTIONS):
                if verbose:
                    print(FUNCTIONS[f].__doc__)
                    print
        else:
            cli = cmd.Cmd()
            cli.columnize(sorted(FUNCTIONS.keys()), displaywidth=80)


def evaluate_line(exp):
    """
    PyParsing core function which parses a string expression.
    """

    # define the parser
    expression = Forward()
    var = Word(alphas)
    lpar = Literal('(').suppress()
    rpar = Literal(')').suppress()
    function = Group(var +
                     lpar +
                     Group(Optional(delimitedList(expression))) +
                     rpar)

    condition = oneOf('< <= > >= == !=')

    integer = Word(nums)
    variable = Word(alphas)
    real = (Combine(Word(nums) + Optional('.' + Word(nums)) +
                    oneOf('e') + Optional(oneOf('+ -')) + Word(nums)) |
            Combine(Word(nums) + '.' + Word(nums))
            )
    node = Combine(Word(alphanums + '_:') + '.' + Word(alphanums + '_[].'))

    operand = function | condition | real | integer | node | variable

    signop = oneOf('+ -')
    multop = oneOf('* / % //')
    plusop = oneOf('+ -')
    expop = Literal('**')

    # use parse actions to attach EvalXXX constructors to sub-expressions
    operand.setParseAction(EvalElement)
    expression << infixNotation(operand,
                                [
                                    (signop, 1, opAssoc.RIGHT, EvalSignOp),
                                    (expop, 2, opAssoc.LEFT, EvalPowerOp),
                                    (multop, 2, opAssoc.LEFT, EvalMultOp),
                                    (plusop, 2, opAssoc.LEFT, EvalAddOp),
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

    # print 'EVAL: %s'%exp
    return ret.eval()


def eval(expression, variables=None):
    """
    Evaluates every line of a given expressions, and handles variable casting when prefixed with "$"
    """
    if not isinstance(expression, (list, tuple)):
        expression = str(expression).splitlines()

    # init known variables    
    known_variables = {}
    if variables:
        known_variables.update(variables)

    result = None
    for line in expression:

        # ignore lines that are commented out, or trailing comments
        line = line.strip()
        if line and not line.startswith('#'):

            if '#' in line:
                line = line.split('#')[0]

            if line.endswith(';'):
                line = line[:-1]

            # --- Process variables --- #
            # Are we piping a result into a variables? (line will begin with $... =)
            stored = None
            list_connect = False
            
            if line.startswith('$'):
                stored = re.findall(r'\$.+?=', line)
                if stored:
                    stored  = stored[0]
                    stored_ = stored[1:-1].strip()

                    if stored_ in known_variables:
                        data = known_variables[stored_]

                        # this will be handeled as a normal variable substitution
                        # in the next phase
                        if isinstance(data, (str,unicode)) and '.' in data:
                            stored = None
                            
                        # if this is a list, we will multi connect to
                        # the result at the very end
                        elif isinstance(data, (list, tuple, set)):
                            list_connect = data
                            line = line.replace(stored, '')
                            stored = None                            
                            
                    # record the network output back into this variable at the very end
                    else:
                        line = line.replace(stored, '')
                        stored = stored[1:-1].strip()
                        
                        
            # else are we giving a list of nodes to connect directly?
            else:
                piped = re.findall('^(.+?)=', line) # watch up to first =
                if piped:
                    # make sure only alphanum with periods and commas
                    if bool(re.match("^[A-Za-z0-9_:|,. -]*$", piped[0])):
                        list_connect = [x.strip() for x in piped[0].split(',')]
                        line = line.replace('%s='%piped[0], '')

            # process known variables and convert any numerics to str
            # and also expand lists
            if known_variables:
                v = sorted([x for x in re.findall("[\d$A-Za-z_]*", line) if x.startswith('$')], key=len)[::-1]
                for var in v:
                    if var[1:] in known_variables:
                        data = known_variables[var[1:]]
                        
                        if isinstance(data, (list, tuple, set)):
                            data = ','.join([str(x) for x in data])
                        else:
                            data = str(data)
                            
                        line = line.replace(var, data)


            # --- evaluate the line --- #
            #print ('eval: %s'%line)
            result = evaluate_line(line)

            # store the result
            if stored:
                known_variables[stored] = result
                
            # list connect?
            if list_connect:
                for item in list_connect:
                    connect(result, item)
                
                
                

    return result

# example usage
# eval('pCube1.t = vector(time1.o, sind(time1.o/360 * 90) * 5, 0)')
# eval('pSphere1.t = choice(int(pCube4.tx), pCube1.t, pCube2.t, pCube3.t)')
