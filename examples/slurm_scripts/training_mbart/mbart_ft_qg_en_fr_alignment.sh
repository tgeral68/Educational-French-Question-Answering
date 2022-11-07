#!/bin/bash
#SBATCH --account=bii@v100
#SBATCH --job-name=mbart_ft_qg_en_fr_alignement
#SBATCH -C v100-32g
#SBATCH --cpus-per-task=80  
#SBATCH --gpus-per-node=4
#SBATCH --time=19:30:00             # temps maximum d'execution demande (HH:MM:SS)
#SBATCH --output=mt5_stdout_%j_%x.out     # nom du fichier de sortie
#SBATCH --error=mt5_stderr_%j_%x.out      # nom du fichier d'erreur (ici commun avec la sortie)
 


source ~/.bashrc
conda activate pytorch-dev
cd $WORK/repositories/Educational-French-Question-Answering/
export EFQADATA=$WORK/LTC/LTC-DATA
export EFQALOG=$WORK/LTC/LTC-LOGS

if [ -z "$1" ]
    then
        srun python -m examples.scripts.mbart_ft_qg --early-stopping-criterion sacrebleu --name mbart_ft_qg_en_fr_alignement --training-set fquad-fr-en.pb.json fquad-fr-fr.pb.json squad-en-en.pb.json squad-en-fr.pb.json piaf-fr-fr.pb.json piaf-fr-en.pb.json  --validation-set fquad-fr-fr.pb.json piaf-fr-fr.pb.json
    else
        srun python -m examples.scripts.mbart_ft_qg --early-stopping-criterion sacrebleu --name mbart_ft_qg_en_fr_alignement --training-set fquad-fr-en.pb.json fquad-fr-fr.pb.json squad-en-en.pb.json squad_en_fr.pb.json piaf-fr-fr.pb.json piaf-fr-en.pb.json  --validation-set fquad-fr-fr.pb.json piaf-fr-fr.pb.json --resume-from-checkpoint $1
fi

