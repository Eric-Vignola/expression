//Maya ASCII 2018 scene
//Name: line_projection.ma
//Last modified: Sun, Apr 11, 2021 01:39:42 PM
//Codeset: 1252
requires maya "2018";
requires "stereoCamera" "10.0";
currentUnit -linear centimeter -angle degree -time ntsc;
fileInfo "application" "maya";
fileInfo "product" "Maya 2018";
fileInfo "version" "2018";
fileInfo "cutIdentifier" "201706261615-f9658c4cfc";
fileInfo "osv" "Microsoft Windows 8 Enterprise Edition, 64-bit  (Build 9200)\n";
createNode transform -shared -name "persp";
	rename -uuid "1ACBFE86-4DF1-8298-D651-F0934350B299";
	setAttr ".visibility" no;
	setAttr ".translate" -type "double3" 7.3551162710907381 17.797743406587699 96.814302182928031 ;
	setAttr ".rotate" -type "double3" -6.9383527296025713 -4.6000000000000307 0 ;
createNode camera -shared -name "perspShape" -parent "persp";
	rename -uuid "569DCD68-4090-88F6-D0A8-EA8834F43E9C";
	setAttr -keyable off ".visibility" no;
	setAttr ".focalLength" 34.999999999999993;
	setAttr ".farClipPlane" 5000;
	setAttr ".centerOfInterest" 98.939272660475552;
	setAttr ".imageName" -type "string" "persp";
	setAttr ".depthName" -type "string" "persp_depth";
	setAttr ".maskName" -type "string" "persp_mask";
	setAttr ".homeCommand" -type "string" "viewSet -p %camera";
	setAttr ".aiTranslator" -type "string" "perspective";
createNode transform -shared -name "top";
	rename -uuid "7C4408A8-4CE4-A32D-BB42-5EB019BAB722";
	setAttr ".visibility" no;
	setAttr ".translate" -type "double3" 0 1000.1 0 ;
	setAttr ".rotate" -type "double3" -89.999999999999986 0 0 ;
createNode camera -shared -name "topShape" -parent "top";
	rename -uuid "1ADA4457-4CE4-272D-2A6B-F4BC41DA4A4A";
	setAttr -keyable off ".visibility" no;
	setAttr ".renderable" no;
	setAttr ".farClipPlane" 5000;
	setAttr ".centerOfInterest" 1000.1;
	setAttr ".orthographicWidth" 30;
	setAttr ".imageName" -type "string" "top";
	setAttr ".depthName" -type "string" "top_depth";
	setAttr ".maskName" -type "string" "top_mask";
	setAttr ".homeCommand" -type "string" "viewSet -t %camera";
	setAttr ".orthographic" yes;
	setAttr ".aiTranslator" -type "string" "orthographic";
createNode transform -shared -name "front";
	rename -uuid "2D9BE2D8-4910-6A1B-2F90-1FA51D43D643";
	setAttr ".visibility" no;
	setAttr ".translate" -type "double3" 0 0 1000.1 ;
createNode camera -shared -name "frontShape" -parent "front";
	rename -uuid "AD4DC56C-4F97-4DA9-1D64-F0B15F1AAE8A";
	setAttr -keyable off ".visibility" no;
	setAttr ".renderable" no;
	setAttr ".farClipPlane" 5000;
	setAttr ".centerOfInterest" 1000.1;
	setAttr ".orthographicWidth" 30;
	setAttr ".imageName" -type "string" "front";
	setAttr ".depthName" -type "string" "front_depth";
	setAttr ".maskName" -type "string" "front_mask";
	setAttr ".homeCommand" -type "string" "viewSet -f %camera";
	setAttr ".orthographic" yes;
	setAttr ".aiTranslator" -type "string" "orthographic";
createNode transform -shared -name "side";
	rename -uuid "D8400D9A-4666-262C-7592-BA8A30BEECF8";
	setAttr ".visibility" no;
	setAttr ".translate" -type "double3" 1000.1 0 0 ;
	setAttr ".rotate" -type "double3" 0 89.999999999999986 0 ;
createNode camera -shared -name "sideShape" -parent "side";
	rename -uuid "17C4E937-4822-F330-486D-6085138C6616";
	setAttr -keyable off ".visibility" no;
	setAttr ".renderable" no;
	setAttr ".farClipPlane" 5000;
	setAttr ".centerOfInterest" 1000.1;
	setAttr ".orthographicWidth" 30;
	setAttr ".imageName" -type "string" "side";
	setAttr ".depthName" -type "string" "side_depth";
	setAttr ".maskName" -type "string" "side_mask";
	setAttr ".homeCommand" -type "string" "viewSet -s %camera";
	setAttr ".orthographic" yes;
	setAttr ".aiTranslator" -type "string" "orthographic";
createNode transform -name "pCube1";
	rename -uuid "4483FEA5-40E3-A7DA-BF30-03AB10D3C705";
	setAttr ".translate" -type "double3" -5.1557421475275431 -8.0519736287735633 0.92903334207487376 ;
createNode mesh -name "pCubeShape1" -parent "pCube1";
	rename -uuid "73FB116B-4D98-1069-7699-4BBF89C09578";
	setAttr -keyable off ".visibility";
	setAttr ".visibleInReflections" yes;
	setAttr ".visibleInRefractions" yes;
	setAttr ".uvSet[0].uvSetName" -type "string" "map1";
	setAttr ".currentUVSet" -type "string" "map1";
	setAttr ".displayColorChannel" -type "string" "Ambient+Diffuse";
	setAttr ".collisionOffsetVelocityMultiplier[0]"  0 1 1;
	setAttr ".collisionDepthVelocityMultiplier[0]"  0 1 1;
	setAttr ".aiTranslator" -type "string" "polymesh";
createNode transform -name "pCube2";
	rename -uuid "03DF6C1E-49DD-4BE9-838F-1581F443A802";
	setAttr ".translate" -type "double3" 15.156755570062234 6.1874136698016287 1.1838479187269542 ;
createNode mesh -name "pCubeShape2" -parent "pCube2";
	rename -uuid "3CA54CF3-4307-A4B2-B089-829FE3DD29DD";
	setAttr -keyable off ".visibility";
	setAttr ".visibleInReflections" yes;
	setAttr ".visibleInRefractions" yes;
	setAttr ".uvSet[0].uvSetName" -type "string" "map1";
	setAttr -size 14 ".uvSet[0].uvSetPoints[0:13]" -type "float2" 0.375
		 0 0.625 0 0.375 0.25 0.625 0.25 0.375 0.5 0.625 0.5 0.375 0.75 0.625 0.75 0.375 1
		 0.625 1 0.875 0 0.875 0.25 0.125 0 0.125 0.25;
	setAttr ".currentUVSet" -type "string" "map1";
	setAttr ".displayColorChannel" -type "string" "Ambient+Diffuse";
	setAttr ".collisionOffsetVelocityMultiplier[0]"  0 1 1;
	setAttr ".collisionDepthVelocityMultiplier[0]"  0 1 1;
	setAttr -size 8 ".vrts[0:7]"  -0.5 -0.5 0.5 0.5 -0.5 0.5 -0.5 0.5 0.5
		 0.5 0.5 0.5 -0.5 0.5 -0.5 0.5 0.5 -0.5 -0.5 -0.5 -0.5 0.5 -0.5 -0.5;
	setAttr -size 12 ".edge[0:11]"  0 1 0 2 3 0 4 5 0 6 7 0 0 2 0 1 3 0
		 2 4 0 3 5 0 4 6 0 5 7 0 6 0 0 7 1 0;
	setAttr -size 6 -capacityHint 24 ".face[0:5]" -type "polyFaces" 
		f 4 0 5 -2 -5
		mu 0 4 0 1 3 2
		f 4 1 7 -3 -7
		mu 0 4 2 3 5 4
		f 4 2 9 -4 -9
		mu 0 4 4 5 7 6
		f 4 3 11 -1 -11
		mu 0 4 6 7 9 8
		f 4 -12 -10 -8 -6
		mu 0 4 1 10 11 3
		f 4 10 4 6 8
		mu 0 4 12 0 2 13;
	setAttr ".creaseData" -type "dataPolyComponent" Index_Data Edge 0 ;
	setAttr ".creaseVertexData" -type "dataPolyComponent" Index_Data Vertex 0 ;
	setAttr ".pinData[0]" -type "dataPolyComponent" Index_Data UV 0 ;
	setAttr ".holeFaceData" -type "dataPolyComponent" Index_Data Face 0 ;
	setAttr ".aiTranslator" -type "string" "polymesh";
createNode transform -name "pCube3";
	rename -uuid "F52F79FD-4BA7-5D19-B282-0AAB710E2EDF";
	setAttr ".translate" -type "double3" -2.0522305252944877 12.23439405149176 -2.1211669040223606 ;
	setAttr -alteredValue ".translateX";
	setAttr -alteredValue ".translateY";
	setAttr -alteredValue ".translateZ";
createNode mesh -name "pCubeShape3" -parent "pCube3";
	rename -uuid "D08868C6-4639-C81B-E6BB-A3B56DEEAB5E";
	setAttr -keyable off ".visibility";
	setAttr ".visibleInReflections" yes;
	setAttr ".visibleInRefractions" yes;
	setAttr ".uvSet[0].uvSetName" -type "string" "map1";
	setAttr -size 14 ".uvSet[0].uvSetPoints[0:13]" -type "float2" 0.375
		 0 0.625 0 0.375 0.25 0.625 0.25 0.375 0.5 0.625 0.5 0.375 0.75 0.625 0.75 0.375 1
		 0.625 1 0.875 0 0.875 0.25 0.125 0 0.125 0.25;
	setAttr ".currentUVSet" -type "string" "map1";
	setAttr ".displayColorChannel" -type "string" "Ambient+Diffuse";
	setAttr ".collisionOffsetVelocityMultiplier[0]"  0 1 1;
	setAttr ".collisionDepthVelocityMultiplier[0]"  0 1 1;
	setAttr -size 8 ".vrts[0:7]"  -0.5 -0.5 0.5 0.5 -0.5 0.5 -0.5 0.5 0.5
		 0.5 0.5 0.5 -0.5 0.5 -0.5 0.5 0.5 -0.5 -0.5 -0.5 -0.5 0.5 -0.5 -0.5;
	setAttr -size 12 ".edge[0:11]"  0 1 0 2 3 0 4 5 0 6 7 0 0 2 0 1 3 0
		 2 4 0 3 5 0 4 6 0 5 7 0 6 0 0 7 1 0;
	setAttr -size 6 -capacityHint 24 ".face[0:5]" -type "polyFaces" 
		f 4 0 5 -2 -5
		mu 0 4 0 1 3 2
		f 4 1 7 -3 -7
		mu 0 4 2 3 5 4
		f 4 2 9 -4 -9
		mu 0 4 4 5 7 6
		f 4 3 11 -1 -11
		mu 0 4 6 7 9 8
		f 4 -12 -10 -8 -6
		mu 0 4 1 10 11 3
		f 4 10 4 6 8
		mu 0 4 12 0 2 13;
	setAttr ".creaseData" -type "dataPolyComponent" Index_Data Edge 0 ;
	setAttr ".creaseVertexData" -type "dataPolyComponent" Index_Data Vertex 0 ;
	setAttr ".pinData[0]" -type "dataPolyComponent" Index_Data UV 0 ;
	setAttr ".holeFaceData" -type "dataPolyComponent" Index_Data Face 0 ;
	setAttr ".aiTranslator" -type "string" "polymesh";
createNode transform -name "pSphere1";
	rename -uuid "CE4909F3-4C01-E3F3-AB22-5684391E4873";
createNode mesh -name "pSphereShape1" -parent "pSphere1";
	rename -uuid "F58039B6-49C2-5656-A264-6390B4937AF2";
	setAttr -keyable off ".visibility";
	setAttr ".visibleInReflections" yes;
	setAttr ".visibleInRefractions" yes;
	setAttr ".uvSet[0].uvSetName" -type "string" "map1";
	setAttr ".currentUVSet" -type "string" "map1";
	setAttr ".displayColorChannel" -type "string" "Ambient+Diffuse";
	setAttr ".collisionOffsetVelocityMultiplier[0]"  0 1 1;
	setAttr ".collisionDepthVelocityMultiplier[0]"  0 1 1;
	setAttr ".aiTranslator" -type "string" "polymesh";
createNode lightLinker -shared -name "lightLinker1";
	rename -uuid "52EE58BD-4757-E305-8096-39A34F89D639";
	setAttr -size 2 ".link";
	setAttr -size 2 ".shadowLink";
createNode shapeEditorManager -name "shapeEditorManager";
	rename -uuid "28B507C8-4A1D-2705-A8F6-89B3BF21CF55";
createNode poseInterpolatorManager -name "poseInterpolatorManager";
	rename -uuid "EBF353D6-4825-B20E-DBC7-499FC8998F5D";
createNode displayLayerManager -name "layerManager";
	rename -uuid "BC50A2F2-4EBB-8F11-3144-F2805CE8E3CC";
createNode displayLayer -name "defaultLayer";
	rename -uuid "732898DC-4748-70A9-DDAD-E798E4878674";
createNode renderLayerManager -name "renderLayerManager";
	rename -uuid "F40CBAD9-4E25-F7F9-3B8E-28AA6DA9DDB4";
createNode renderLayer -name "defaultRenderLayer";
	rename -uuid "51F7C295-41D8-D102-5EBE-C8BDEA794DF0";
	setAttr ".global" yes;
createNode polyCube -name "polyCube1";
	rename -uuid "9EF69193-417C-CFDF-D3A7-01AC1E75D6A2";
	setAttr ".createUVs" 4;
createNode polySphere -name "polySphere1";
	rename -uuid "9BFB754D-455D-FC4E-0B24-0BBA8E869A8B";
createNode plusMinusAverage -name "plusMinusAverage1";
	rename -uuid "3930EFBE-471A-8AA9-E537-DF889DF729C5";
	setAttr -lock on ".operation" 2;
	setAttr -size 2 ".input3D";
	setAttr -size 2 ".input3D";
createNode distanceBetween -name "distanceBetween1";
	rename -uuid "6D929C8F-45AD-E49C-256F-D69D93233686";
createNode multiplyDivide -name "multiplyDivide1";
	rename -uuid "8398ED6D-4F69-D13E-710C-309ABE4858DA";
createNode network -name "constant1";
	rename -uuid "8A9D7E42-427A-E84F-66FF-958A40707DAF";
	addAttr -cachedInternally true -keyable true -shortName "value" -longName "value" 
		-attributeType "double";
	setAttr -keyable on ".value";
createNode network -name "constant2";
	rename -uuid "849541F6-4350-6EBF-E9EE-0FBE0C063BCE";
	addAttr -cachedInternally true -keyable true -shortName "value" -longName "value" 
		-defaultValue 2 -attributeType "double";
	setAttr -keyable on ".value";
createNode condition -name "condition1";
	rename -uuid "7ED70416-4028-84F1-B5F3-C28EEF0BF74F";
	setAttr -lock on ".operation";
createNode network -name "vector1";
	rename -uuid "19591B8B-41D3-6F9A-981F-BFA863CB54BB";
	addAttr -cachedInternally true -keyable true -shortName "value" -longName "value" 
		-attributeType "double3" -numberOfChildren 3;
	addAttr -cachedInternally true -keyable true -shortName "valueX" -longName "valueX" 
		-attributeType "double" -parent "value";
	addAttr -cachedInternally true -keyable true -shortName "valueY" -longName "valueY" 
		-attributeType "double" -parent "value";
	addAttr -cachedInternally true -keyable true -shortName "valueZ" -longName "valueZ" 
		-attributeType "double" -parent "value";
	setAttr -keyable on ".value";
	setAttr -keyable on ".valueX";
	setAttr -keyable on ".valueY";
	setAttr -keyable on ".valueZ";
createNode condition -name "condition2";
	rename -uuid "0348FB18-4D54-7A18-C5FD-47B198C85E17";
	setAttr -lock on ".operation";
createNode condition -name "condition3";
	rename -uuid "80175A7A-42A3-48CE-DD8C-DE9BCE6EE1AC";
	setAttr -lock on ".operation";
createNode condition -name "condition4";
	rename -uuid "99451132-4765-9B5D-DF1F-378E5A59AF0F";
	setAttr -lock on ".operation";
createNode plusMinusAverage -name "plusMinusAverage2";
	rename -uuid "692418A2-4709-EF7A-F9E8-4380DAE730FD";
	setAttr -lock on ".operation" 2;
	setAttr -size 2 ".input3D";
	setAttr -size 2 ".input3D";
createNode vectorProduct -name "vectorProduct1";
	rename -uuid "5955E053-4385-FA29-0A90-90B02B8E360C";
	setAttr -lock on ".operation";
	setAttr -lock on ".normalizeOutput";
createNode multiplyDivide -name "multiplyDivide2";
	rename -uuid "7182673A-4583-87EC-8271-BA88A38D55FB";
	setAttr -lock on ".operation";
createNode plusMinusAverage -name "plusMinusAverage3";
	rename -uuid "A3E5048D-4331-2991-7585-A09045B5BEDF";
	setAttr -lock on ".operation";
	setAttr -size 2 ".input3D";
	setAttr -size 2 ".input3D";
createNode network -name "constant3";
	rename -uuid "36390BA6-4829-53CF-2AD0-F281A95E34F2";
	addAttr -cachedInternally true -keyable true -shortName "value" -longName "value" 
		-attributeType "double";
	setAttr -keyable on ".value";
createNode network -name "vector2";
	rename -uuid "BC54194C-4E38-1C60-0CDC-48AAB53C02BB";
	addAttr -cachedInternally true -keyable true -shortName "value" -longName "value" 
		-attributeType "double3" -numberOfChildren 3;
	addAttr -cachedInternally true -keyable true -shortName "valueX" -longName "valueX" 
		-attributeType "double" -parent "value";
	addAttr -cachedInternally true -keyable true -shortName "valueY" -longName "valueY" 
		-attributeType "double" -parent "value";
	addAttr -cachedInternally true -keyable true -shortName "valueZ" -longName "valueZ" 
		-attributeType "double" -parent "value";
	setAttr -keyable on ".value";
	setAttr -keyable on ".valueX";
	setAttr -keyable on ".valueY";
	setAttr -keyable on ".valueZ";
createNode condition -name "condition5";
	rename -uuid "A8CB3A79-489F-E570-5EF6-8194E7DBBB71";
	setAttr -lock on ".operation" 4;
createNode condition -name "condition6";
	rename -uuid "C108C251-42EC-6E5C-D883-0E8994CEE28A";
	setAttr -lock on ".operation" 4;
createNode condition -name "condition7";
	rename -uuid "06305B26-45AE-3445-A1D8-EC933F31DF6B";
	setAttr -lock on ".operation" 4;
createNode distanceBetween -name "distanceBetween2";
	rename -uuid "3168FA20-4D63-227A-6311-C3A04360C396";
createNode network -name "vector3";
	rename -uuid "38D3B498-4C2D-8A72-C947-509C10ED3354";
	addAttr -cachedInternally true -keyable true -shortName "value" -longName "value" 
		-attributeType "double3" -numberOfChildren 3;
	addAttr -cachedInternally true -keyable true -shortName "valueX" -longName "valueX" 
		-attributeType "double" -parent "value";
	addAttr -cachedInternally true -keyable true -shortName "valueY" -longName "valueY" 
		-attributeType "double" -parent "value";
	addAttr -cachedInternally true -keyable true -shortName "valueZ" -longName "valueZ" 
		-attributeType "double" -parent "value";
	setAttr -keyable on ".value";
	setAttr -keyable on ".valueX";
	setAttr -keyable on ".valueY";
	setAttr -keyable on ".valueZ";
createNode condition -name "condition8";
	rename -uuid "8C4DAF51-4C37-744A-A3B6-63AF575D0B88";
	setAttr -lock on ".operation" 2;
createNode condition -name "condition9";
	rename -uuid "0441D8EC-473F-AC27-5A70-76AB1287E95B";
	setAttr -lock on ".operation" 2;
createNode condition -name "condition10";
	rename -uuid "8E983495-43B8-75BE-2017-56B59B789C37";
	setAttr -lock on ".operation" 2;
createNode script -name "sceneConfigurationScriptNode";
	rename -uuid "344F6E3E-4C01-5AE0-AC3C-51B2724C5464";
	setAttr ".before" -type "string" "playbackOptions -min 0 -max 100 -ast 0 -aet 100 ";
	setAttr ".scriptType" 6;
createNode animCurveTL -name "pCube3_translateX";
	rename -uuid "E8EB0515-463D-6947-EF93-13A9EA8FC67B";
	setAttr ".tangentType" 18;
	setAttr ".weightedTangents" no;
	setAttr -size 2 ".keyTimeValue[0:1]"  0 -7.5412109526656073 62 18.365416865971831;
createNode animCurveTL -name "pCube3_translateY";
	rename -uuid "C662DC59-4702-EF80-D56B-E7B5A6BDC401";
	setAttr ".tangentType" 18;
	setAttr ".weightedTangents" no;
	setAttr -size 2 ".keyTimeValue[0:1]"  0 4.6986407857915324 62 27.293007285235518;
createNode animCurveTL -name "pCube3_translateZ";
	rename -uuid "DB6D4FC4-4DDC-BC4A-96AD-9CA665DE494E";
	setAttr ".tangentType" 18;
	setAttr ".weightedTangents" no;
	setAttr -size 2 ".keyTimeValue[0:1]"  0 -1.8963542632718742 62 -2.5704075423175397;
createNode nodeGraphEditorInfo -name "MayaNodeEditorSavedTabsInfo";
	rename -uuid "41E748B4-40AF-C40B-D330-9791B864FEF7";
	setAttr ".tabGraphInfo[0].tabName" -type "string" "Untitled_1";
	setAttr ".tabGraphInfo[0].viewRectLow" -type "double2" -2782.148839516125 -1645.8491967803775 ;
	setAttr ".tabGraphInfo[0].viewRectHigh" -type "double2" 2595.9158444928285 1309.99374725178 ;
	setAttr -size 35 ".tabGraphInfo[0].nodeInfo";
	setAttr ".tabGraphInfo[0].nodeInfo[0].positionX" -1712.857177734375;
	setAttr ".tabGraphInfo[0].nodeInfo[0].positionY" 285.71429443359375;
	setAttr ".tabGraphInfo[0].nodeInfo[0].nodeVisualState" 18304;
	setAttr ".tabGraphInfo[0].nodeInfo[1].positionX" 744.28570556640625;
	setAttr ".tabGraphInfo[0].nodeInfo[1].positionY" 750;
	setAttr ".tabGraphInfo[0].nodeInfo[1].nodeVisualState" 18304;
	setAttr ".tabGraphInfo[0].nodeInfo[2].positionX" 1358.5714111328125;
	setAttr ".tabGraphInfo[0].nodeInfo[2].positionY" 778.5714111328125;
	setAttr ".tabGraphInfo[0].nodeInfo[2].nodeVisualState" 18304;
	setAttr ".tabGraphInfo[0].nodeInfo[3].positionX" 1358.5714111328125;
	setAttr ".tabGraphInfo[0].nodeInfo[3].positionY" 677.14288330078125;
	setAttr ".tabGraphInfo[0].nodeInfo[3].nodeVisualState" 18304;
	setAttr ".tabGraphInfo[0].nodeInfo[4].positionX" 2047.142822265625;
	setAttr ".tabGraphInfo[0].nodeInfo[4].positionY" -392.85714721679688;
	setAttr ".tabGraphInfo[0].nodeInfo[4].nodeVisualState" 18304;
	setAttr ".tabGraphInfo[0].nodeInfo[5].positionX" -1712.857177734375;
	setAttr ".tabGraphInfo[0].nodeInfo[5].positionY" 488.57144165039063;
	setAttr ".tabGraphInfo[0].nodeInfo[5].nodeVisualState" 18304;
	setAttr ".tabGraphInfo[0].nodeInfo[6].positionX" -791.4285888671875;
	setAttr ".tabGraphInfo[0].nodeInfo[6].positionY" 395.71429443359375;
	setAttr ".tabGraphInfo[0].nodeInfo[6].nodeVisualState" 18304;
	setAttr ".tabGraphInfo[0].nodeInfo[7].positionX" -2327.142822265625;
	setAttr ".tabGraphInfo[0].nodeInfo[7].positionY" 508.57144165039063;
	setAttr ".tabGraphInfo[0].nodeInfo[7].nodeVisualState" 18304;
	setAttr ".tabGraphInfo[0].nodeInfo[8].positionX" 744.28570556640625;
	setAttr ".tabGraphInfo[0].nodeInfo[8].positionY" 648.5714111328125;
	setAttr ".tabGraphInfo[0].nodeInfo[8].nodeVisualState" 18304;
	setAttr ".tabGraphInfo[0].nodeInfo[9].positionX" 1665.7142333984375;
	setAttr ".tabGraphInfo[0].nodeInfo[9].positionY" 778.5714111328125;
	setAttr ".tabGraphInfo[0].nodeInfo[9].nodeVisualState" 18304;
	setAttr ".tabGraphInfo[0].nodeInfo[10].positionX" 1051.4285888671875;
	setAttr ".tabGraphInfo[0].nodeInfo[10].positionY" 850;
	setAttr ".tabGraphInfo[0].nodeInfo[10].nodeVisualState" 18304;
	setAttr ".tabGraphInfo[0].nodeInfo[11].positionX" 1432.857177734375;
	setAttr ".tabGraphInfo[0].nodeInfo[11].positionY" -392.85714721679688;
	setAttr ".tabGraphInfo[0].nodeInfo[11].nodeVisualState" 18304;
	setAttr ".tabGraphInfo[0].nodeInfo[12].positionX" -484.28570556640625;
	setAttr ".tabGraphInfo[0].nodeInfo[12].positionY" 420;
	setAttr ".tabGraphInfo[0].nodeInfo[12].nodeVisualState" 18304;
	setAttr ".tabGraphInfo[0].nodeInfo[13].positionX" -484.28570556640625;
	setAttr ".tabGraphInfo[0].nodeInfo[13].positionY" 650;
	setAttr ".tabGraphInfo[0].nodeInfo[13].nodeVisualState" 18304;
	setAttr ".tabGraphInfo[0].nodeInfo[14].positionX" 437.14285278320313;
	setAttr ".tabGraphInfo[0].nodeInfo[14].positionY" 715.71429443359375;
	setAttr ".tabGraphInfo[0].nodeInfo[14].nodeVisualState" 18304;
	setAttr ".tabGraphInfo[0].nodeInfo[15].positionX" 1972.857177734375;
	setAttr ".tabGraphInfo[0].nodeInfo[15].positionY" 778.5714111328125;
	setAttr ".tabGraphInfo[0].nodeInfo[15].nodeVisualState" 18304;
	setAttr ".tabGraphInfo[0].nodeInfo[16].positionX" 1358.5714111328125;
	setAttr ".tabGraphInfo[0].nodeInfo[16].positionY" 880;
	setAttr ".tabGraphInfo[0].nodeInfo[16].nodeVisualState" 18304;
	setAttr ".tabGraphInfo[0].nodeInfo[17].positionX" -1098.5714111328125;
	setAttr ".tabGraphInfo[0].nodeInfo[17].positionY" 107.14286041259766;
	setAttr ".tabGraphInfo[0].nodeInfo[17].nodeVisualState" 18304;
	setAttr ".tabGraphInfo[0].nodeInfo[18].positionX" -2327.142822265625;
	setAttr ".tabGraphInfo[0].nodeInfo[18].positionY" 407.14285278320313;
	setAttr ".tabGraphInfo[0].nodeInfo[18].nodeVisualState" 18304;
	setAttr ".tabGraphInfo[0].nodeInfo[19].positionX" 744.28570556640625;
	setAttr ".tabGraphInfo[0].nodeInfo[19].positionY" 547.14288330078125;
	setAttr ".tabGraphInfo[0].nodeInfo[19].nodeVisualState" 18304;
	setAttr ".tabGraphInfo[0].nodeInfo[20].positionX" -2020;
	setAttr ".tabGraphInfo[0].nodeInfo[20].positionY" 414.28570556640625;
	setAttr ".tabGraphInfo[0].nodeInfo[20].nodeVisualState" 18304;
	setAttr ".tabGraphInfo[0].nodeInfo[21].positionX" 130;
	setAttr ".tabGraphInfo[0].nodeInfo[21].positionY" 525.71429443359375;
	setAttr ".tabGraphInfo[0].nodeInfo[21].nodeVisualState" 18304;
	setAttr ".tabGraphInfo[0].nodeInfo[22].positionX" -1405.7142333984375;
	setAttr ".tabGraphInfo[0].nodeInfo[22].positionY" 270;
	setAttr ".tabGraphInfo[0].nodeInfo[22].nodeVisualState" 18304;
	setAttr ".tabGraphInfo[0].nodeInfo[23].positionX" -791.4285888671875;
	setAttr ".tabGraphInfo[0].nodeInfo[23].positionY" 717.14288330078125;
	setAttr ".tabGraphInfo[0].nodeInfo[23].nodeVisualState" 18304;
	setAttr ".tabGraphInfo[0].nodeInfo[24].positionX" -1712.857177734375;
	setAttr ".tabGraphInfo[0].nodeInfo[24].positionY" 387.14285278320313;
	setAttr ".tabGraphInfo[0].nodeInfo[24].nodeVisualState" 18304;
	setAttr ".tabGraphInfo[0].nodeInfo[25].positionX" 1051.4285888671875;
	setAttr ".tabGraphInfo[0].nodeInfo[25].positionY" 691.4285888671875;
	setAttr ".tabGraphInfo[0].nodeInfo[25].nodeVisualState" 18304;
	setAttr ".tabGraphInfo[0].nodeInfo[26].positionX" -791.4285888671875;
	setAttr ".tabGraphInfo[0].nodeInfo[26].positionY" 192.85714721679688;
	setAttr ".tabGraphInfo[0].nodeInfo[26].nodeVisualState" 18304;
	setAttr ".tabGraphInfo[0].nodeInfo[27].positionX" 437.14285278320313;
	setAttr ".tabGraphInfo[0].nodeInfo[27].positionY" 500;
	setAttr ".tabGraphInfo[0].nodeInfo[27].nodeVisualState" 18304;
	setAttr ".tabGraphInfo[0].nodeInfo[28].positionX" -791.4285888671875;
	setAttr ".tabGraphInfo[0].nodeInfo[28].positionY" 294.28570556640625;
	setAttr ".tabGraphInfo[0].nodeInfo[28].nodeVisualState" 18304;
	setAttr ".tabGraphInfo[0].nodeInfo[29].positionX" -177.14285278320313;
	setAttr ".tabGraphInfo[0].nodeInfo[29].positionY" 645.71429443359375;
	setAttr ".tabGraphInfo[0].nodeInfo[29].nodeVisualState" 18304;
	setAttr ".tabGraphInfo[0].nodeInfo[30].positionX" 1740;
	setAttr ".tabGraphInfo[0].nodeInfo[30].positionY" -392.85714721679688;
	setAttr ".tabGraphInfo[0].nodeInfo[30].nodeVisualState" 18304;
	setAttr ".tabGraphInfo[0].nodeInfo[31].positionX" -187.14285278320313;
	setAttr ".tabGraphInfo[0].nodeInfo[31].positionY" -1030;
	setAttr ".tabGraphInfo[0].nodeInfo[31].nodeVisualState" 18304;
	setAttr ".tabGraphInfo[0].nodeInfo[32].positionX" -1098.5714111328125;
	setAttr ".tabGraphInfo[0].nodeInfo[32].positionY" 868.5714111328125;
	setAttr ".tabGraphInfo[0].nodeInfo[32].nodeVisualState" 18304;
	setAttr ".tabGraphInfo[0].nodeInfo[33].positionX" -1098.5714111328125;
	setAttr ".tabGraphInfo[0].nodeInfo[33].positionY" 767.14288330078125;
	setAttr ".tabGraphInfo[0].nodeInfo[33].nodeVisualState" 18304;
	setAttr ".tabGraphInfo[0].nodeInfo[34].positionX" -1098.5714111328125;
	setAttr ".tabGraphInfo[0].nodeInfo[34].positionY" 665.71429443359375;
	setAttr ".tabGraphInfo[0].nodeInfo[34].nodeVisualState" 18304;
select -noExpand :time1;
	setAttr ".outTime" 24;
	setAttr ".unwarpedTime" 24;
select -noExpand :hardwareRenderingGlobals;
	setAttr ".objectTypeFilterNameArray" -type "stringArray" 22 "NURBS Curves" "NURBS Surfaces" "Polygons" "Subdiv Surface" "Particles" "Particle Instance" "Fluids" "Strokes" "Image Planes" "UI" "Lights" "Cameras" "Locators" "Joints" "IK Handles" "Deformers" "Motion Trails" "Components" "Hair Systems" "Follicles" "Misc. UI" "Ornaments"  ;
	setAttr ".objectTypeFilterValueArray" -type "Int32Array" 22 0 1 1
		 1 1 1 1 1 1 0 0 0 0 0 0
		 0 0 0 0 0 0 0 ;
	setAttr ".floatingPointRTEnable" yes;
select -noExpand :renderPartition;
	setAttr -size 2 ".sets";
select -noExpand :renderGlobalsList1;
select -noExpand :defaultShaderList1;
	setAttr -size 4 ".shaders";
select -noExpand :postProcessList1;
	setAttr -size 2 ".postProcesses";
select -noExpand :defaultRenderingList1;
select -noExpand :initialShadingGroup;
	setAttr -size 4 ".dagSetMembers";
	setAttr ".renderableOnlySet" yes;
select -noExpand :initialParticleSE;
	setAttr ".renderableOnlySet" yes;
select -noExpand :defaultRenderGlobals;
	setAttr ".comFrrt" 30;
	setAttr ".currentRenderer" -type "string" "arnold";
	setAttr ".startFrame" 1;
	setAttr ".endFrame" 10;
select -noExpand :defaultResolution;
	setAttr ".pixelAspect" 1;
select -noExpand :hardwareRenderGlobals;
	setAttr ".colorTextureResolution" 256;
	setAttr ".bumpTextureResolution" 512;
	setAttr ".hardwareFrameRate" 30;
connectAttr "polyCube1.output" "pCubeShape1.inMesh";
connectAttr "pCube3_translateX.output" "pCube3.translateX";
connectAttr "pCube3_translateY.output" "pCube3.translateY";
connectAttr "pCube3_translateZ.output" "pCube3.translateZ";
connectAttr "vector3.value" "pSphere1.translate";
connectAttr "polySphere1.output" "pSphereShape1.inMesh";
relationship "link" ":lightLinker1" ":initialShadingGroup.message" ":defaultLightSet.message";
relationship "link" ":lightLinker1" ":initialParticleSE.message" ":defaultLightSet.message";
relationship "shadowLink" ":lightLinker1" ":initialShadingGroup.message" ":defaultLightSet.message";
relationship "shadowLink" ":lightLinker1" ":initialParticleSE.message" ":defaultLightSet.message";
connectAttr "layerManager.displayLayerId[0]" "defaultLayer.identification";
connectAttr "renderLayerManager.renderLayerId[0]" "defaultRenderLayer.identification"
		;
connectAttr "pCube2.translate" "plusMinusAverage1.input3D[0]";
connectAttr "pCube1.translate" "plusMinusAverage1.input3D[1]";
connectAttr "plusMinusAverage1.output3D" "distanceBetween1.point1";
connectAttr "plusMinusAverage1.output3D" "multiplyDivide1.input1";
connectAttr "distanceBetween1.distance" "multiplyDivide1.input2X";
connectAttr "distanceBetween1.distance" "multiplyDivide1.input2Y";
connectAttr "distanceBetween1.distance" "multiplyDivide1.input2Z";
connectAttr "condition1.outColorB" "multiplyDivide1.operation";
connectAttr "distanceBetween1.distance" "condition1.firstTerm";
connectAttr "constant1.value" "condition1.secondTerm";
connectAttr "constant1.value" "condition1.colorIfTrueR";
connectAttr "constant1.value" "condition1.colorIfTrueG";
connectAttr "constant1.value" "condition1.colorIfTrueB";
connectAttr "constant2.value" "condition1.colorIfFalseR";
connectAttr "constant2.value" "condition1.colorIfFalseG";
connectAttr "constant2.value" "condition1.colorIfFalseB";
connectAttr "condition2.outColorB" "vector1.valueX";
connectAttr "condition3.outColorB" "vector1.valueY";
connectAttr "condition4.outColorB" "vector1.valueZ";
connectAttr "distanceBetween1.distance" "condition2.firstTerm";
connectAttr "constant1.value" "condition2.secondTerm";
connectAttr "condition1.outColorR" "condition2.colorIfTrueR";
connectAttr "condition1.outColorR" "condition2.colorIfTrueG";
connectAttr "condition1.outColorR" "condition2.colorIfTrueB";
connectAttr "multiplyDivide1.outputX" "condition2.colorIfFalseR";
connectAttr "multiplyDivide1.outputX" "condition2.colorIfFalseG";
connectAttr "multiplyDivide1.outputX" "condition2.colorIfFalseB";
connectAttr "distanceBetween1.distance" "condition3.firstTerm";
connectAttr "constant1.value" "condition3.secondTerm";
connectAttr "condition1.outColorG" "condition3.colorIfTrueR";
connectAttr "condition1.outColorG" "condition3.colorIfTrueG";
connectAttr "condition1.outColorG" "condition3.colorIfTrueB";
connectAttr "multiplyDivide1.outputY" "condition3.colorIfFalseR";
connectAttr "multiplyDivide1.outputY" "condition3.colorIfFalseG";
connectAttr "multiplyDivide1.outputY" "condition3.colorIfFalseB";
connectAttr "distanceBetween1.distance" "condition4.firstTerm";
connectAttr "constant1.value" "condition4.secondTerm";
connectAttr "condition1.outColorB" "condition4.colorIfTrueR";
connectAttr "condition1.outColorB" "condition4.colorIfTrueG";
connectAttr "condition1.outColorB" "condition4.colorIfTrueB";
connectAttr "multiplyDivide1.outputZ" "condition4.colorIfFalseR";
connectAttr "multiplyDivide1.outputZ" "condition4.colorIfFalseG";
connectAttr "multiplyDivide1.outputZ" "condition4.colorIfFalseB";
connectAttr "pCube3.translate" "plusMinusAverage2.input3D[0]";
connectAttr "pCube1.translate" "plusMinusAverage2.input3D[1]";
connectAttr "plusMinusAverage2.output3D" "vectorProduct1.input1";
connectAttr "vector1.value" "vectorProduct1.input2";
connectAttr "vectorProduct1.output" "multiplyDivide2.input1";
connectAttr "vector1.value" "multiplyDivide2.input2";
connectAttr "multiplyDivide2.output" "plusMinusAverage3.input3D[0]";
connectAttr "pCube1.translate" "plusMinusAverage3.input3D[1]";
connectAttr "condition5.outColorB" "vector2.valueX";
connectAttr "condition6.outColorB" "vector2.valueY";
connectAttr "condition7.outColorB" "vector2.valueZ";
connectAttr "vectorProduct1.outputX" "condition5.firstTerm";
connectAttr "constant3.value" "condition5.secondTerm";
connectAttr "pCube1.translateX" "condition5.colorIfTrueR";
connectAttr "pCube1.translateX" "condition5.colorIfTrueG";
connectAttr "pCube1.translateX" "condition5.colorIfTrueB";
connectAttr "plusMinusAverage3.output3Dx" "condition5.colorIfFalseR";
connectAttr "plusMinusAverage3.output3Dx" "condition5.colorIfFalseG";
connectAttr "plusMinusAverage3.output3Dx" "condition5.colorIfFalseB";
connectAttr "vectorProduct1.outputY" "condition6.firstTerm";
connectAttr "constant3.value" "condition6.secondTerm";
connectAttr "pCube1.translateY" "condition6.colorIfTrueR";
connectAttr "pCube1.translateY" "condition6.colorIfTrueG";
connectAttr "pCube1.translateY" "condition6.colorIfTrueB";
connectAttr "plusMinusAverage3.output3Dy" "condition6.colorIfFalseR";
connectAttr "plusMinusAverage3.output3Dy" "condition6.colorIfFalseG";
connectAttr "plusMinusAverage3.output3Dy" "condition6.colorIfFalseB";
connectAttr "vectorProduct1.outputZ" "condition7.firstTerm";
connectAttr "constant3.value" "condition7.secondTerm";
connectAttr "pCube1.translateZ" "condition7.colorIfTrueR";
connectAttr "pCube1.translateZ" "condition7.colorIfTrueG";
connectAttr "pCube1.translateZ" "condition7.colorIfTrueB";
connectAttr "plusMinusAverage3.output3Dz" "condition7.colorIfFalseR";
connectAttr "plusMinusAverage3.output3Dz" "condition7.colorIfFalseG";
connectAttr "plusMinusAverage3.output3Dz" "condition7.colorIfFalseB";
connectAttr "plusMinusAverage1.output3D" "distanceBetween2.point1";
connectAttr "condition8.outColorB" "vector3.valueX";
connectAttr "condition9.outColorB" "vector3.valueY";
connectAttr "condition10.outColorB" "vector3.valueZ";
connectAttr "vectorProduct1.outputX" "condition8.firstTerm";
connectAttr "distanceBetween2.distance" "condition8.secondTerm";
connectAttr "pCube2.translateX" "condition8.colorIfTrueR";
connectAttr "pCube2.translateX" "condition8.colorIfTrueG";
connectAttr "pCube2.translateX" "condition8.colorIfTrueB";
connectAttr "vector2.valueX" "condition8.colorIfFalseR";
connectAttr "vector2.valueX" "condition8.colorIfFalseG";
connectAttr "vector2.valueX" "condition8.colorIfFalseB";
connectAttr "vectorProduct1.outputY" "condition9.firstTerm";
connectAttr "distanceBetween2.distance" "condition9.secondTerm";
connectAttr "pCube2.translateY" "condition9.colorIfTrueR";
connectAttr "pCube2.translateY" "condition9.colorIfTrueG";
connectAttr "pCube2.translateY" "condition9.colorIfTrueB";
connectAttr "vector2.valueY" "condition9.colorIfFalseR";
connectAttr "vector2.valueY" "condition9.colorIfFalseG";
connectAttr "vector2.valueY" "condition9.colorIfFalseB";
connectAttr "vectorProduct1.outputZ" "condition10.firstTerm";
connectAttr "distanceBetween2.distance" "condition10.secondTerm";
connectAttr "pCube2.translateZ" "condition10.colorIfTrueR";
connectAttr "pCube2.translateZ" "condition10.colorIfTrueG";
connectAttr "pCube2.translateZ" "condition10.colorIfTrueB";
connectAttr "vector2.valueZ" "condition10.colorIfFalseR";
connectAttr "vector2.valueZ" "condition10.colorIfFalseG";
connectAttr "vector2.valueZ" "condition10.colorIfFalseB";
connectAttr "constant2.message" "MayaNodeEditorSavedTabsInfo.tabGraphInfo[0].nodeInfo[0].dependNode"
		;
connectAttr "condition6.message" "MayaNodeEditorSavedTabsInfo.tabGraphInfo[0].nodeInfo[1].dependNode"
		;
connectAttr "condition8.message" "MayaNodeEditorSavedTabsInfo.tabGraphInfo[0].nodeInfo[2].dependNode"
		;
connectAttr "condition9.message" "MayaNodeEditorSavedTabsInfo.tabGraphInfo[0].nodeInfo[3].dependNode"
		;
connectAttr ":initialShadingGroup.message" "MayaNodeEditorSavedTabsInfo.tabGraphInfo[0].nodeInfo[4].dependNode"
		;
connectAttr "distanceBetween1.message" "MayaNodeEditorSavedTabsInfo.tabGraphInfo[0].nodeInfo[5].dependNode"
		;
connectAttr "condition2.message" "MayaNodeEditorSavedTabsInfo.tabGraphInfo[0].nodeInfo[6].dependNode"
		;
connectAttr "pCube2.message" "MayaNodeEditorSavedTabsInfo.tabGraphInfo[0].nodeInfo[7].dependNode"
		;
connectAttr "condition7.message" "MayaNodeEditorSavedTabsInfo.tabGraphInfo[0].nodeInfo[8].dependNode"
		;
connectAttr "vector3.message" "MayaNodeEditorSavedTabsInfo.tabGraphInfo[0].nodeInfo[9].dependNode"
		;
connectAttr "distanceBetween2.message" "MayaNodeEditorSavedTabsInfo.tabGraphInfo[0].nodeInfo[10].dependNode"
		;
connectAttr "polySphere1.message" "MayaNodeEditorSavedTabsInfo.tabGraphInfo[0].nodeInfo[11].dependNode"
		;
connectAttr "vector1.message" "MayaNodeEditorSavedTabsInfo.tabGraphInfo[0].nodeInfo[12].dependNode"
		;
connectAttr "plusMinusAverage2.message" "MayaNodeEditorSavedTabsInfo.tabGraphInfo[0].nodeInfo[13].dependNode"
		;
connectAttr "constant3.message" "MayaNodeEditorSavedTabsInfo.tabGraphInfo[0].nodeInfo[14].dependNode"
		;
connectAttr "pSphere1.message" "MayaNodeEditorSavedTabsInfo.tabGraphInfo[0].nodeInfo[15].dependNode"
		;
connectAttr "condition10.message" "MayaNodeEditorSavedTabsInfo.tabGraphInfo[0].nodeInfo[16].dependNode"
		;
connectAttr "multiplyDivide1.message" "MayaNodeEditorSavedTabsInfo.tabGraphInfo[0].nodeInfo[17].dependNode"
		;
connectAttr "pCube1.message" "MayaNodeEditorSavedTabsInfo.tabGraphInfo[0].nodeInfo[18].dependNode"
		;
connectAttr "condition5.message" "MayaNodeEditorSavedTabsInfo.tabGraphInfo[0].nodeInfo[19].dependNode"
		;
connectAttr "plusMinusAverage1.message" "MayaNodeEditorSavedTabsInfo.tabGraphInfo[0].nodeInfo[20].dependNode"
		;
connectAttr "multiplyDivide2.message" "MayaNodeEditorSavedTabsInfo.tabGraphInfo[0].nodeInfo[21].dependNode"
		;
connectAttr "condition1.message" "MayaNodeEditorSavedTabsInfo.tabGraphInfo[0].nodeInfo[22].dependNode"
		;
connectAttr "pCube3.message" "MayaNodeEditorSavedTabsInfo.tabGraphInfo[0].nodeInfo[23].dependNode"
		;
connectAttr "constant1.message" "MayaNodeEditorSavedTabsInfo.tabGraphInfo[0].nodeInfo[24].dependNode"
		;
connectAttr "vector2.message" "MayaNodeEditorSavedTabsInfo.tabGraphInfo[0].nodeInfo[25].dependNode"
		;
connectAttr "condition3.message" "MayaNodeEditorSavedTabsInfo.tabGraphInfo[0].nodeInfo[26].dependNode"
		;
connectAttr "plusMinusAverage3.message" "MayaNodeEditorSavedTabsInfo.tabGraphInfo[0].nodeInfo[27].dependNode"
		;
connectAttr "condition4.message" "MayaNodeEditorSavedTabsInfo.tabGraphInfo[0].nodeInfo[28].dependNode"
		;
connectAttr "vectorProduct1.message" "MayaNodeEditorSavedTabsInfo.tabGraphInfo[0].nodeInfo[29].dependNode"
		;
connectAttr "pSphereShape1.message" "MayaNodeEditorSavedTabsInfo.tabGraphInfo[0].nodeInfo[30].dependNode"
		;
connectAttr "sceneConfigurationScriptNode.message" "MayaNodeEditorSavedTabsInfo.tabGraphInfo[0].nodeInfo[31].dependNode"
		;
connectAttr "pCube3_translateX.message" "MayaNodeEditorSavedTabsInfo.tabGraphInfo[0].nodeInfo[32].dependNode"
		;
connectAttr "pCube3_translateY.message" "MayaNodeEditorSavedTabsInfo.tabGraphInfo[0].nodeInfo[33].dependNode"
		;
connectAttr "pCube3_translateZ.message" "MayaNodeEditorSavedTabsInfo.tabGraphInfo[0].nodeInfo[34].dependNode"
		;
connectAttr "defaultRenderLayer.message" ":defaultRenderingList1.rendering" -nextAvailable
		;
connectAttr "pCubeShape1.instObjGroups" ":initialShadingGroup.dagSetMembers" -nextAvailable
		;
connectAttr "pCubeShape2.instObjGroups" ":initialShadingGroup.dagSetMembers" -nextAvailable
		;
connectAttr "pCubeShape3.instObjGroups" ":initialShadingGroup.dagSetMembers" -nextAvailable
		;
connectAttr "pSphereShape1.instObjGroups" ":initialShadingGroup.dagSetMembers" -nextAvailable
		;
// End of line_projection.ma
