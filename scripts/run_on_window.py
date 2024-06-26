# evo_prot_grad/scripts/run_on_window.py
'''
* Author: Evan Komp
* Created: 6/17/2024
* Company: Bottle Institute @ National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT

Generates variants according to ESM likelihood for sets of fixed positions determined to be important elsewhere.

For each scaffold and each max_depth specified, <batch_size> positions are sampled according to their weights.
For each of those samples, <variants_per_batch> variants are generated.

INPUTS:
- `scaffold_table` (str): Path to the scaffold table
    This is a csv file with `id`, `sequence`, and `position_weight` columns.
    The sequence is the wild type sequence.
    The position weight is a string listing positions that are important, their current AA, and their importance weight of the form '<position>:<AA>:<weight>;<position>:<AA>:<weight>;...'.
    Positions are 1 indexed.
    For example, a row in the dataframe might look like:
    ```
    id,sequence,position_weight
    1,MAKLSKJF,"1:A:0.4,2:K:0.3,3:L:0.2"
    ```
- `sequence_id_subset` (List[str]): List of sequence ids to generate variants for
- `output_path` (str): Path to the output file
- `esm_model` (str): Name of ESM model to use
- `max_depths` (List[int]): List of number of positions to allow to vary
- `batch_size` (int): Number of weighted samples per scaffold per depth
- `variants_per_batch` (int): Number of variants to generate per batch

EXAMPLE:
input sequence = "MAKLSKJF"
position_weight = "1:A:0.4,2:K:0.3,3:L:0.2"
max_depths = [1, 2, 3]
batch_size = 5
variants_per_batch = 3

OUTPUT:
for max_depth = 1:
   sample 1 positions from position_weight
   generate 3 variants
   repeat 5 total times
for max_depth = 2:
    sample 2 positions from position_weight
    ...

Total variants generated = 3 * 5 * 3 = 45
'''
import argparse

import os

import pandas as pd
import numpy as np
from tqdm import tqdm

import evo_prot_grad
from transformers import AutoTokenizer, EsmForMaskedLM

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', filemode='w', filename='run_on_window.log')

def get_n_mutations(sequence, variant):
    """Get number of mutations between two sequences.

    Args:
        sequence (str): Wild type sequence
        variant (str): Variant sequence
    Returns:
        int: Number of mutations
    """
    return sum(aa1.lower() != aa2.lower() for aa1, aa2 in zip(sequence, variant))

def parse_position_weights(position_weight, sequence):
    """Parse position weights into a dictionary.
    
    Input is of the form '1:A:0.4;2:K:0.3;3:L:0.2'
    Convert to a dictionary of the form {0: 0.4, 1: 0.3, 2: 0.2}
    Makes it 0 indexed as expected by EvoProtGrad.
    Also checks that AA at position is correct.
    Also norma lizes weights.

    Args:
        position_weight (str): Position weight string
        sequence (str): Wild type sequence
    Returns:
        dict: Dictionary of position weights
    """
    position_weights = {}
    for pos_aa_weight in position_weight.split(';'):
        if pos_aa_weight == '':
            continue
        pos, aa, weight = pos_aa_weight.split(':')
        pos = int(pos) - 1
        assert sequence[pos] == aa, f"AA at position {pos} is not {aa} as expected."
        position_weights[pos] = float(weight)

    # normalize weights
    sum_weight = sum(position_weights.values())
    for pos in position_weights:
        position_weights[pos] /= sum_weight
    return position_weights

def variable_positions_to_fixed_positions(positions, sequence_length):
    """Convert variable positions to fixed positions.
    
    EvoProtGrad expects lists of fixed position ranges of the form [(start1, end1), (start2, end2), ...]
    A single Aa fixed position is represented as (start, end) where start = end.

    Convert a list of variable positions to a list of fixed position ranges.
    """
    indices_set = set(positions)
    all_indices = set(range(sequence_length))
    other_indices = sorted(all_indices - indices_set)

    ranges = []
    if other_indices:
        start = end = other_indices[0]
        for i in other_indices[1:]:
            if i == end + 1:
                end = i
            else:
                ranges.append((start, end))
                start = end = i
        ranges.append((start, end))

    return ranges    

def generate_variants(expert_list, sequence, sequence_id, max_depth, batch_size, variants_per_batch, position_weights):
    """For a sequence, generate variants at positions sampled from position_weights.

    Args:
        sampler (Sampler): Sampler object from EvoProtGrad
        sequence (str): Wild type sequence
        sequence_id (str): Sequence identifier
        max_depth (int): Number of positions to vary
        batch_size (int): Number of samples to generate
        variants_per_batch (int): Number of variants to generate per sample
        position_weights (dict): Dictionary of position weights
    """
    for batch in range(batch_size):
        # select positions to allow to vary
        positions_chosen = np.random.choice(list(position_weights.keys()), size=max_depth, replace=False, p=list(position_weights.values()))
        fixed_positions = variable_positions_to_fixed_positions(positions_chosen, len(sequence))

        # generate variants
        sampler = evo_prot_grad.DirectedEvolution(
            wt_protein=sequence,
            experts=expert_list,
            preserved_regions=fixed_positions,
            max_mutations=max_depth,
            parallel_chains=variants_per_batch,
            n_steps=500, # 10000 used in the paper
            verbose=False,
            output='best'
        )
        variants, scores = sampler()

        # capitalize mutated positions
        new_variants = []
        for variant in variants:
            new_variant = variant.lower().split()
            for pos in positions_chosen:
                if new_variant[pos] != sequence[pos].lower():
                    new_variant[pos] = new_variant[pos].upper()
                else:
                    new_variant[pos] = new_variant[pos].lower()
            new_variants.append(''.join(new_variant))
        variants = new_variants


        n_mutations = [get_n_mutations(sequence, variant) for variant in variants]
        positions = [';'.join(positions_chosen.astype(str))] * len(variants)
        ids = [f"{sequence_id}_{max_depth}_{batch}_{i}" for i in range(len(variants))]
        scaffolds = [sequence_id] * len(variants)
        yield pd.DataFrame({'scaffold': scaffolds, 'id': ids, 'sequence': variants, 'n_mutations': n_mutations, 'positions': positions, 'score': scores})

def main(args):

    scaffold_table = pd.read_csv(args.scaffold_table)
    assert 'id' in scaffold_table, "id column not found in scaffold table."
    assert 'sequence' in scaffold_table, "sequence column not found in scaffold table."
    assert 'position_weight' in scaffold_table, "position_weight column not found in scaffold table."
    scaffold_table['position_weight'] = scaffold_table.apply(lambda row: parse_position_weights(row['position_weight'], row['sequence']), axis=1)

    subset = args.sequence_id_subset
    if subset:
        scaffold_table = scaffold_table[scaffold_table['id'].isin(subset)]

    print(scaffold_table)
    
    esm_expert = evo_prot_grad.get_expert(
        expert_name='esm',
        temperature=1.0,
        device='cuda',
        model=EsmForMaskedLM.from_pretrained(args.esm_model),
        tokenizer=AutoTokenizer.from_pretrained(args.esm_model),
        scoring_strategy='mutant_marginal'
        
    )
    expert_list = [esm_expert]

    max_depths = args.max_depths
    batch_size = args.batch_size
    variants_per_batch = args.variants_per_batch

    max_calls = len(scaffold_table) * len(max_depths) * batch_size
    max_variants = max_calls * variants_per_batch
    logging.info(f"Generating up to {max_variants} variants.")
    with open(args.output_path, 'w') as f:
        # write to output one batch at a time
        f.write('scaffold,id,sequence,n_mutations,positions,score\n')
        with tqdm(total=max_calls, desc="Sampler calls") as pbar:
            for _, row in scaffold_table.iterrows():
                sequence = row['sequence']
                sequence_id = row['id']
                position_weights = row['position_weight']
                for max_depth in max_depths:
                    for df in generate_variants(expert_list, sequence, sequence_id, max_depth, batch_size, variants_per_batch, position_weights):
                        df.to_csv(f, header=False, index=False)
                        logging.info(f"Generated {len(df)} variants for {sequence_id} at max_depth {max_depth}")
                        pbar.update(1)
    # open up the file and remove any duplicate sequences
    logging.info(f"Generated a total of {len(df)} sequences")
    df = pd.read_csv(args.output_path)
    df = df.drop_duplicates(subset=['sequence', 'scaffold'])
    logging.info(f"After dropping duplicates: {len(df)} sequences")
    # also drop 0 mutations
    df = df[df['n_mutations']>0]
    logging.info(f"After dropping WT repeats: {len(df)} sequences")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scaffold_table', type=str, required=True)
    parser.add_argument('--sequence_id_subset', type=str, nargs='+', required=False)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--esm_model', type=str, required=True)
    parser.add_argument('--max_depths', type=int, nargs='+', required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--variants_per_batch', type=int, required=True)
    args = parser.parse_args()

    main(args)
