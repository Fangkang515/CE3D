run-all:
	python3 chat_edit_3D.py --port 7862 --clean_FBends --load "Segmenting_cuda:0,ImageCaptioning_cuda:0,Text2Image_cuda:0,VisualQuestionAnswering_cuda:0,Text2Box_cuda:0,Inpainting_cuda:0,InstructPix2Pix_cuda:0,Image2Depth_cuda:0,DepthText2Image_cuda:0,SRImage_cuda:0,Image2Scribble_cuda:1,ScribbleText2Image_cuda:1,Image2Canny_cuda:1,CannyText2Image_cuda:1,Image2Line_cuda:1,LineText2Image_cuda:1,Image2Hed_cuda:1,HedText2Image_cuda:1,Image2Pose_cuda:1,PoseText2Image_cuda:1,SegText2Image_cuda:1,Image2Normal_cuda:1,NormalText2Image_cuda:1,ReferenceImageTextEditing_cuda:1"

run-small-instruct:
	python3 chat_edit_3D.py --port 7862 --clean_FBends --load "Segmenting_cuda:0,ImageCaptioning_cuda:0,VisualQuestionAnswering_cuda:0,Text2Box_cuda:0,Inpainting_cuda:0,InstructPix2Pix_cuda:0"






