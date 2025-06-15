from rdkit import Chem
import random
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


# create a function to generate augmented SMILES
def generate_augmented_smiles(smiles, max_attempts=10, random_state=42):
    random.seed(random_state)
    dict_augmented = {}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"WARNING: RDKit could not parse SMILES: {smiles}")
        return dict_augmented  # empty dict
    for i in range(max_attempts):
        augmented = Chem.MolToSmiles(mol, canonical=False, doRandom=True)
        dict_augmented[i] = augmented
    return dict_augmented


# generate random reaction smiles
def generate_random_reaction_smiles(reaction_smiles, max_attempts=10, random_state=42):
    random.seed(random_state)
    dict_reactions = {}
    # break reaction smiles into reactants and products
    reactants, agents, products = reaction_smiles.split('>')
    # generate augmented smiles for reactants, agents, and products
    dict_reactants = generate_augmented_smiles(reactants, max_attempts=max_attempts, random_state=random_state)
    dict_agents = generate_augmented_smiles(agents, max_attempts=max_attempts, random_state=random_state)
    dict_products = generate_augmented_smiles(products, max_attempts=max_attempts, random_state=random_state)
    # Check for empty dicts to avoid KeyError
    if not dict_reactants or not dict_agents or not dict_products:
        print(f"WARNING: Could not generate augmented SMILES for: {reaction_smiles}")
        # return the original reaction smiles
        return {0: reaction_smiles}
    # generate random reaction smiles
    for i in range(max_attempts):
        reactant_rnd = random.randint(0, max_attempts-1)
        agent_rnd = random.randint(0, max_attempts-1)
        product_rnd = random.randint(0, max_attempts-1)
        reaction = dict_reactants[reactant_rnd] + '>' + dict_agents[agent_rnd] + '>' + dict_products[product_rnd]
        dict_reactions[i] = reaction
    return dict_reactions
    
    
    
