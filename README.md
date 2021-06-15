# Expression
Expression to math network parser for Autodesk Maya

## About
This port of a MEL library I wrote decades ago to convert a string expression into a math node networks.
Parsing the expression is now handled by pyparsing, but the math tree logic remains and is 100% native
Maya math nodes, no plugins required.

The main aim of this module is to solve the problem scripted rig pipelines have where several paragraphs of
`createNode` and `connectAttr` commands can now be replaced by something human readable, like `pCube2.t = pCube1.t * floor(pCube3.ty)`.

## Requirements
Autodesk Maya, pyparsing

## Author
* **Eric Vignola** (eric.vignola@gmail.com)

## Example
A simple line that will make a cube surf over time
```
eval('pCube1.t = vector(time1.o, sind(time1.o/360 * 90) * 5, 0)')
```

You can sequentially call the command and reuse the output:
```
# ye olde line projection example
# (maya file in the examples folder)
import expression

segment = expression.eval('pCube2.t - pCube1.t')           # line segment defined between pCube1 and pCube2
unit    = expression.eval('unit(%s)'%segment)              # unit vector of the segment
vector  = expression.eval('pCube3.t - pCube1.t')           # vector between pCube3 and pCube1 for projection
dot     = expression.eval('dot(%s,%s)'%(vector,unit))      # dot product of vector over unit segment
proj    = expression.eval('%s * %s + pCube1.t'%(dot,unit)) # projection over the line segment

# clamp at start if dot < 0, if True position is pCube1.t, if False position is projection
lesser  = expression.eval('if(%s < 0, pCube1.t, %s)'%(dot,proj))

# clamp at end if dot > mag(segment), if True position is pCube2.t, if False position is projection
# plug the result to pSphere1.t
expression.eval('pSphere1.t = if(%s > mag(%s), pCube2.t, %s)'%(dot,segment,lesser)) 

```

Or you can use basic variable substitution in a multiline expression:
```
import expression

exp = '''
$SEGMENT = $CUBE2.t - $CUBE1.t     # line segment defined between pCube1 and pCube2
$UNIT    = unit($SEGMENT)          # unit vector of the segment
$VECTOR  = $CUBE3.t - $CUBE1.t     # vector between pCube3 and pCube1 for projection
$DOT     = dot($VECTOR, $UNIT)     # dot product of vector over unit segment
$PROJ    = $DOT * $UNIT + $CUBE1.t # projection over the line segment

# clamp at start if dot < 0, if True position is pCube1.t, if False position is projection
$LESSER  = if($DOT < 0, $CUBE1.t, $PROJ)

# clamp at end if dot > mag(segment), if True position is pCube2.t, if False position is projection
# plug the result to pSphere1.t
$SPHERE.t = if($DOT > mag($SEGMENT), $CUBE2.t, $LESSER)
'''

CUBE1  = 'pCube1'
CUBE2  = 'pCube2'
CUBE3  = 'pCube3'
SPHERE = 'pSphere1'

expression.eval(exp, variables=globals())
```

## Supported Functions
```
abs       floor                  pointMatrixProduct
acos      frame                  radians
acosd     if                     rev
asin      int                    sign
asind     inv                    sin
avg       mag                    sind
ceil      matrixMultiply         sum
clamp     max                    tan
cos       min                    tand
cosd      nCross                 unit
cross     nDot                   vector
degrees   nPointMatrixProduct    vectorMatrixProduct           
dist      nVectorMatrixProduct   
dot       noise
easeIn    
easeOut    
```


## License
BSD 3-Clause License:
Copyright (c)  2021, Eric Vignola 
All rights reserved. 

Redistribution and use in source and binary forms, with or without 
modification, are permitted provided that the following conditions are met:


1. Redistributions of source code must retain the above copyright notice, 
   this list of conditions and the following disclaimer.
   
2. Redistributions in binary form must reproduce the above copyright notice, 
   this list of conditions and the following disclaimer in the documentation 
   and/or other materials provided with the distribution.
   
3. Neither the name of copyright holders nor the names of its 
   contributors may be used to endorse or promote products derived from 
   this software without specific prior written permission.
   
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE 
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, 
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

