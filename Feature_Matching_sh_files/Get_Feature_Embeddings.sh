#BSUB -o Get_Feature_embeddings_model_2_Test.%J
#BSUB -N
#BSUB -J Get_Feature_embeddings_model_2_Test
#BSUB -gpu "num=4:mode=exclusive_process:gmodel=NVIDIAGeForceGTX1080Ti"
#BSUB -q normal
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/research/c.shuting/anaconda3/lib
python ../code_Feature_Matching/main_get_emb.py