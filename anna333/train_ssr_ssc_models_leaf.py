import pandas as pd
import numpy as np
import os
from utils import FastaSequenceLoader, ConvNetwork
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
mapped_read_counts = [
    'zea_counts.csv',
    'solanum_counts.csv',
    'arabidopsis_counts.csv',
    'sbicolor_counts.csv',
    'Oryza_sativa_counts.csv',
    'Actinidia_chinensis_counts.csv',
    'Aegilops_tauschii_counts.csv',
    'Arabis_alpina_counts.csv',
    'Beta_vulgaris_counts.csv',
    'Brachypodium_distachyon_counts.csv',
    'Brassica_juncea_counts.csv',
    'Brassica_oleracea_counts.csv',
    'Brassica_rapa_counts.csv',
    'Camelina_sativa_counts.csv',
    'Physcomitrium_patens_counts.csv',
]

gene_models = [
    'Zea_mays.Zm-B73-REFERENCE-NAM-5.0.52.gtf',
    'Solanum_lycopersicum.SL3.0.52.gtf',
    'Arabidopsis_thaliana.TAIR10.52.gtf',
    'Sorghum_bicolor.Sorghum_bicolor_NCBIv3.52.gtf',
    'Oryza_sativa.IRGSP-1.0.61.gtf',
    'Actinidia_chinensis.Red5_PS1_1.69.0.61.gtf',
    'Aegilops_tauschii.Aet_v4.0.61.gtf',
    'Arabis_alpina.A_alpina_V4.61.gtf',
    'Beta_vulgaris.RefBeet-1.2.2.61.gtf',
    'Brachypodium_distachyon.Brachypodium_distachyon_v3.0.61.gtf',
    'Brassica_juncea.ASM1870372v1.61.gtf',
    'Brassica_oleracea.BOL.61.gtf',
    'Brassica_rapa.Brapa_1.0.61.gtf',
    'Camelina_sativa.Cs.61.gtf',
    'Physcomitrium_patens.Phypa_V3.61.gtf',
]

genomes = [
    'Zea_mays.Zm-B73-REFERENCE-NAM-5.0.dna.toplevel.fa',
    'Solanum_lycopersicum.SL3.0.dna.toplevel.fa',
    'Arabidopsis_thaliana.TAIR10.dna.toplevel.fa',
    'Sorghum_bicolor.Sorghum_bicolor_NCBIv3.dna.toplevel.fa',
    'Oryza_sativa.IRGSP-1.0.dna.toplevel.fa',
    'Actinidia_chinensis.Red5_PS1_1.69.0.dna.toplevel.fa',
    'Aegilops_tauschii.Aet_v4.0.dna.toplevel.fa',
    'Arabis_alpina.A_alpina_V4.dna.toplevel.fa',
    'Beta_vulgaris.RefBeet-1.2.2.dna.toplevel.fa',
    'Brachypodium_distachyon.Brachypodium_distachyon_v3.0.dna.toplevel.fa',
    'Brassica_juncea.ASM1870372v1.dna.toplevel.fa',
    'Brassica_oleracea.BOL.dna.toplevel.fa',
    'Brassica_rapa.Brapa_1.0.dna.toplevel.fa',
    'Camelina_sativa.Cs.dna.toplevel.fa',
    'Physcomitrium_patens.Phypa_V3.dna.toplevel.fa',
]

pickle_keys = [
    'zea', 'sol', 'ara', 'sor',
    'osa', 'Act', 'Aet', 'Aal',
    'Bvu', 'Bdi', 'Bju', 'Bol',
    'Bra', 'Csa', 'Ppa',
]

num_chromosomes = [
    10, 12, 5, 10,
    12,  # Oryza
    29,  # Actinidia
    7,  # Aegilops #
    8,  # Arabis
    9,  # Beta #
    5,  # Brachypodium #
    10,  # Brassica juncea #
    9,  # Brassica oleracea
    10,  # Brassica rapa #
    20,  # Camelina #
    27,  # Physcomitrium
]

if not os.path.isdir('../results'):
    os.mkdir('../results')
if not os.path.isdir('saved_models'):
    os.mkdir('saved_models')

for m_reads, gene_model, genome, num_chr, p_key in zip(mapped_read_counts, gene_models, genomes, num_chromosomes,
                                                       pickle_keys):
    if not os.path.exists(f"../results/{m_reads.split('_')[0]}_result.csv"):
        final_training_output = []
        tpm_counts = pd.read_csv(f'tpm_counts/{m_reads}', index_col=0)
        true_targets = []

        for log_count in tpm_counts['logMaxTPM'].values:
            if log_count <= np.percentile(tpm_counts['logMaxTPM'], 25):
                true_targets.append(0)
            elif log_count >= np.percentile(tpm_counts['logMaxTPM'], 75):
                true_targets.append(1)
            else:
                true_targets.append(2)
        tpm_counts['true_target'] = true_targets
        print(tpm_counts.head())

        for val_chromosome in np.arange(1, num_chr+1):
            fastaloader = FastaSequenceLoader(f'genomes/{genome}', f'gene_models/{gene_model}',
                                              val_chromosome, pickled_val_ids='validation_genes.pickle',
                                              pickled_key=p_key)
            enc_train, enc_val, train_ids, val_ids = fastaloader.extract_seq()

            print('-----------------------------------------------------------------------------\n')
            print(f"Plant: {m_reads.split('_')[0]} Case: promoter_terminator")
            print('-------------------------------------------------------------------------------')
            convnet = ConvNetwork(enc_train, enc_val, train_ids, val_ids, val_chromosome, tpm_counts,
                                  m_reads.split('_')[0], 'promoter_terminator')
            output = convnet.train_network()
            final_training_output.append(output)

            # Train models with shuffled sequences
            print('-----------------------------------------------------------------------------\n')
            print(f"Plant: {m_reads.split('_')[0]} Case: si-nucleotide_shuffle")
            print('-------------------------------------------------------------------------------')
            shuffle_enc_train = []
            for train_seq in enc_train.copy():
                np.random.shuffle(train_seq)
                shuffle_enc_train.append(train_seq)

            shuffle_convnet = ConvNetwork(shuffle_enc_train, enc_val, train_ids, val_ids, val_chromosome, tpm_counts,
                                          m_reads.split('_')[0], 'si-nucleotide_shuffle')
            shuffle_output = shuffle_convnet.train_network()
            final_training_output.append(shuffle_output)

        final_training_output = pd.DataFrame(final_training_output, columns=['val_acc', 'val_auROC', 'plant', 'case',
                                                                             'training size'])
        final_training_output.to_csv(f"../results/{m_reads.split('_')[0]}_leaf_result.csv", index=False)

