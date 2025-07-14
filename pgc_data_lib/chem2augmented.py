from rdkit import Chem
from rdkit.Chem import AllChem
import random
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


# create a function to generate augmented SMILES
def generate_augmented_smiles(smiles, max_attempts=10, random_state=42):

    if smiles == '':
        return {0: smiles}

    random.seed(random_state)
    dict_augmented = {}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"WARNING: RDKit could not parse SMILES: {smiles}")
        return dict_augmented  # empty dict
    for i in range(max_attempts):
        augmented = Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=True)  # isomericSmiles=True to preserve stereochemistry !!!
        dict_augmented[i] = augmented
    return dict_augmented


# generate random reaction smiles
def generate_random_reaction_smiles(reaction_smiles, max_attempts=10, random_state=42, product_canonical=True):
    random.seed(random_state)
    dict_reactions = {}
    
    # Check if reaction_smiles has the expected format with two '>' characters
    if reaction_smiles.count('>') != 2:
        print(f"WARNING: Reaction SMILES does not have the expected format (missing '>'): {reaction_smiles}")
        return {0: reaction_smiles}  # Return original as fallback
    
    # Try parsing as a reaction first
    try:
        rxn = AllChem.ReactionFromSmarts(reaction_smiles, useSmiles=True)
        if rxn is None:
            print(f"WARNING: RDKit could not parse reaction SMILES: {reaction_smiles}")
            return {0: reaction_smiles}  # Return original as fallback
    except Exception as e:
        print(f"WARNING: Error parsing reaction SMILES: {reaction_smiles}, error: {str(e)}")
        return {0: reaction_smiles}  # Return original as fallback
        
    # break reaction smiles into reactants and products
    try:
        reactants, agents, products = reaction_smiles.split('>')
    except ValueError:
        print(f"WARNING: Could not split reaction SMILES correctly: {reaction_smiles}")
        return {0: reaction_smiles}  # Return original as fallback
    # generate augmented smiles for reactants, agents, and products
    dict_reactants = generate_augmented_smiles(reactants, max_attempts=max_attempts, random_state=random_state)
    dict_agents = generate_augmented_smiles(agents, max_attempts=max_attempts, random_state=random_state)
    if product_canonical:
        # create only one entry, which is canonical
        dict_products = {0: Chem.MolToSmiles(Chem.MolFromSmiles(products), canonical=True)}
        # mol = Chem.MolFromSmiles(products)
        # if mol is None:
        #     print(f"WARNING: RDKit could not parse product SMILES: {products}")
        #     dict_products = {0: products}  # fallback to original
        # else:
        #     dict_products = {0: Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)}
    else:
        dict_products = generate_augmented_smiles(products, max_attempts=max_attempts, random_state=random_state)
    # Check for empty dicts to avoid KeyError
    if not dict_reactants or not dict_products:  # Agents can be empty
        print(f"WARNING: Could not generate augmented SMILES for: {reaction_smiles}")
        # return the original reaction smiles
        return {0: reaction_smiles}
    # generate random reaction smiles
    for i in range(max_attempts):
        reactant_rnd = random.randint(0, max_attempts-1)
        # agent_rnd can be 0 if there is no agent
        if agents == '':
            agent_rnd = 0
        else:
            agent_rnd = random.randint(0, max_attempts-1)   
        if product_canonical:
            product_rnd = 0
        else:
            product_rnd = random.randint(0, max_attempts-1)
        reaction = dict_reactants[reactant_rnd] + '>' + dict_agents[agent_rnd] + '>' + dict_products[product_rnd]
        dict_reactions[i] = reaction
    return dict_reactions
    
    
    
