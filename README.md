# frontiers_code


To train a model use the main.py function. 
The most important arguments are : 
    - --segmentation-only to train only on the segmentation task
    - --registration-only to train only on the registration task
    - --with-loss-trick to add the seg^0 (Equation 4 of the article)
    - --only-t1 to train only with t1 modality (by default 4 modality are excepted)
    - --source-target--merge-operation ['subtraction', 'addition', 'concatenation'] to choose the mergin operation used
    
To train the proposed method, the commands line is :

python -m frontiers_brain.main --only-t1 --session-name seg_reg_SubMerge_8channels_1.0ratio_0.002lr --epochs 180 --batch-size 2 --lr 2e-3 --nb-gpu 1 --only-brats --source-target-merge-operation subtraction --n-channels-first-layer 8 --ratio-weights-registration-over-segmentation 1.0 -deform-regularisation 1e-10

To do the inference, use the inference.py function. 
The most important arguments are :
  - --get-segmentation to save the predicted segmentation
  - --get-registration to save the predicted registration
  - --model-abspath to give the absolue path of one model to do the inference with
  - --models-folder to give the absolue path of one folder containing different models to do the inference with
  
To do the inference, the command line is : 

python -m frontiers_brain_inference.inference  --get-registration --data-folder-path path_to_data --model-abspath path_to_model --output-folder path_to_output
