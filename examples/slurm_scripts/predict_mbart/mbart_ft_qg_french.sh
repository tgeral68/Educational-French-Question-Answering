#!/bin/bash
#SBATCH --account=bii@v100
#SBATCH --job-name=mbart_ft_qg_en_fr_alignement
#SBATCH -C v100-32g
#SBATCH --cpus-per-task=20  
#SBATCH --gpus-per-node=1
#SBATCH --time=05:00:00             # temps maximum d'execution demande (HH:MM:SS)
#SBATCH --output=mt5_stdout_%j_%x.out     # nom du fichier de sortie
#SBATCH --error=mt5_stderr_%j_%x.out      # nom du fichier d'erreur (ici commun avec la sortie)
 


source ~/.bashrc
conda activate pytorch-dev
cd $WORK/repositories/Educational-French-Question-Answering/
export EFQADATA=$WORK/LTC/LTC-DATA
export EFQALOG=$WORK/LTC/LTC-LOGS



srun python -m examples.scripts.mbart_pred_qg --from-checkpoint $EFQALOG/mbart_ft_qg_french/version_0/checkpoints/val-rouge-checkpoint-epoch=02-val_rouge=0.00.ckpt --test-dataset-folder $EFQADATA/test_dataset --batch-size 16
