from util.tokenizer import EmptyProgramException
from util.c_tokenizer import C_Tokenizer
from util.helpers import get_rev_dict, make_dir_if_not_exists as mkdir, remove_empty_new_lines
import os, time, argparse, sqlite3, json, numpy as np
from functools import partial
from data_processing.typo_mutator import LoopCountThresholdExceededException, FailedToMutateException, Typo_Mutate, typo_mutate
import pandas as pd

def generate_name_dict_store(db_path, bins):
    # #print db_path
    name_dict_store={}
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        for row in cursor.execute("SELECT code_id, name_dict, name_seq FROM Code;"):
            code_id = str(row[0])
            name_dict = json.loads(row[1])
            name_seq = json.loads(row[2])
            name_dict_store[code_id] = (name_dict, name_seq)
    #print 'name_dict_store len:', len(name_dict_store)
    return name_dict_store


# maintain max_fix_length to keep consistentcy with deepfix.
def generate_training_data(db_path, bins, validation_users, min_program_length, max_program_length, \
                                    max_fix_length, max_mutations, max_variants, seed):
    rng = np.random.RandomState(seed)
    tokenize = C_Tokenizer().tokenize
    convert_to_new_line_format = C_Tokenizer().convert_to_new_line_format

    mutator_obj = Typo_Mutate(rng)
    # print(mutator_obj)
    mutate = partial(typo_mutate, mutator_obj)
    # print(mutate)

    token_strings = {'train': {}, 'validation': {}}

    exceptions_in_mutate_call = 0
    total_mutate_calls = 0
    program_lengths, fix_lengths = [], []

    problem_list = []
    for bin_ in bins:
        for problem_id in bin_:
            problem_list.append(problem_id)

    # problem_list = ['prob200']
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        query = "SELECT user_id, code_id, tokenized_code FROM Code " + "WHERE problem_id=? and codelength>? and codelength<? and errorcount=0;"
        count = 0

        problem_id_count = 0
        code_ids = []

        for problem_id in problem_list:
            problem_id_count += 1
            code_id_count = 0

            for row in cursor.execute(query, (problem_id, min_program_length, max_program_length)):
                user_id, code_id, tokenized_program = map(str, row)
                if code_id:  #== 'prog42036':

                    code_id_count += 1
                    code_ids.append(code_id)
                    key = 'validation' if user_id in validation_users[problem_id] else 'train'  # how validation users decided

                    program_length = len(tokenized_program.split())
                    program_lengths.append(program_length)

                    if min_program_length <= program_length <= max_program_length:
                        # Mutate
                        total_mutate_calls += 1
                        try:
                            # print(max_mutations)  # what is it?
                            # print(max_variants)     # what is it?
                            iterator = mutate(tokenized_program, max_mutations, max_variants)

                        except FailedToMutateException:
                            exceptions_in_mutate_call += 1
                        except LoopCountThresholdExceededException:
                            exceptions_in_mutate_call += 1
                        except ValueError:
                            exceptions_in_mutate_call += 1
                            raise
                        except AssertionError:
                            exceptions_in_mutate_call += 1
                            raise
                        except Exception:
                            exceptions_in_mutate_call += 1
                            raise
                        else:
                            tokenized_program = remove_empty_new_lines(convert_to_new_line_format(tokenized_program))

                            for corrupt_program, fix in iterator:
                                corrupt_program_length = len(corrupt_program.split())
                                fix_length             = len(fix.split())
                                fix_lengths.append(fix_length)

                                if corrupt_program_length >= min_program_length and \
                                corrupt_program_length <= max_program_length and fix_length <= max_fix_length:

                                    corrupt_program = remove_empty_new_lines(convert_to_new_line_format(corrupt_program))
                                    try:
                                        token_strings[key][problem_id] += [(code_id, corrupt_program, tokenized_program)]
                                    except:
                                        token_strings[key][problem_id] = [(code_id, corrupt_program, tokenized_program)]

        program_lengths = np.sort(program_lengths)
    fix_lengths = np.sort(fix_lengths)

    print(problem_id_count)
    print(len(code_ids))

    df = pd.DataFrame(code_ids)
    df.to_csv("code_ids_1.csv")

    print('Statistics')
    print('----------')
    print('Program length:  Mean =', np.mean(program_lengths), '\t95th %ile =', program_lengths[int(0.95 * len(program_lengths))])

    try:
        print('Mean fix length: Mean =', np.mean(fix_lengths), '\t95th %ile = ', fix_lengths[int(0.95 * len(fix_lengths))])
    except Exception as e:
        print(e)
        print('fix_lengths')
        print(fix_lengths)
    print('Total mutate calls:', total_mutate_calls)
    print('Exceptions in mutate() call:', exceptions_in_mutate_call, '\n')

    for key in token_strings:
        #print key
        for problem_id in token_strings[key]:
            print (problem_id, len(token_strings[key][problem_id]))

    return token_strings, mutator_obj.get_mutation_distribution()


def build_dictionary(token_strings, tldict={}):

    def build_dict(list_generator, dict_ref):
        for tokenized_program in list_generator:
            for token in tokenized_program.split():
                token = token.strip()
                if token not in dict_ref:
                    dict_ref[token] = len(dict_ref)

    tldict['_pad_'] = 0
    tldict['_eos_'] = 1
    tldict['-new-line-'] = 2

    for key in token_strings:
        for problem_id in token_strings[key]:
            build_dict( ( corr_prog for _, inc_prog, corr_prog in token_strings[key][problem_id]), tldict)

    # required for some programs in the test dataset.
    for idx in range(33):
        if '_<id>_%d@' % idx not in tldict:
            tldict['_<id>_%d@' % idx] = len(tldict)

    print('dictionary size:', len(tldict))
    assert len(tldict) > 50
    return tldict


def vectorize(tokens, tldict, max_vector_length):
    vec_tokens = []
    for token in tokens.split():
        try:
            vec_tokens.append(tldict[token])
        except Exception:
            #print token
            raise

    if len(vec_tokens) > max_vector_length:
        return None

    return vec_tokens


def vectorize_data(token_strings, tldict, max_program_length):
    token_vectors = {}
    skipped = 0

    for key in token_strings:
        token_vectors[key] = {}
        for problem_id in token_strings[key]:
            token_vectors[key][problem_id] = []

    for key in token_strings:
        for problem_id in token_strings[key]:
            for code_id, prog_tokens, fix_tokens in token_strings[key][problem_id]:
                inc_prog_vector = vectorize(prog_tokens, tldict, max_program_length)
                corr_prog_vector = vectorize(fix_tokens, tldict,  max_program_length)

                if (inc_prog_vector is not None) and (corr_prog_vector is not None):
                    token_vectors[key][problem_id].append((code_id, inc_prog_vector, corr_prog_vector))
                else:
                    skipped += 1

    print('skipped during vectorization:', skipped)
    return token_vectors

def save_dictionaries(destination, tldict):
    all_dicts = (tldict, get_rev_dict(tldict))
    np.save(os.path.join(destination, 'all_dicts.npy'), all_dicts)

def load_dictionaries(destination):
    tldict, rev_tldict = np.load(os.path.join(destination, 'all_dicts.npy'), allow_pickle=True)
    return tldict, rev_tldict


def save_pairs(destination, token_vectors, tldict, name_dict_store):
    np.save(os.path.join(destination, ('name_dict_store.npy')), name_dict_store )
    save_dictionaries(destination, tldict)
    for key in token_vectors.keys():
        np.save(os.path.join(destination, ('examples-%s.npy' % key)), token_vectors[key] )


def save_bins(destination, tldict, token_vectors, bins, name_dict_store):
    full_list = []
    for bin_ in bins:
        for problem_id in bin_:
            full_list.append(problem_id)

    print("full_list", len(full_list))

    for i, bin_ in enumerate(bins):
        test_problems = bin_
        training_problems = list(set(full_list) - set(bin_))

        token_vectors_this_fold = { 'train': {}, 'validation': {}, 'test': {} }

        for problem_id in training_problems:
            if problem_id in token_vectors['train']:
                for code_id, inc_prog_vector, corr_prog_vector in token_vectors['train'][problem_id]:
                    variant = 1
                    temp_code_id = code_id
                    while temp_code_id in token_vectors_this_fold['train']:
                        temp_code_id = code_id + '_v%d' % variant
                        variant += 1
                    variant = 1
                    code_id = temp_code_id

                    token_vectors_this_fold['train'][code_id] = (inc_prog_vector, corr_prog_vector)

                if problem_id in token_vectors['validation']:
                    for code_id, inc_prog_vector, corr_prog_vector in token_vectors['validation'][problem_id]:
                        variant = 1
                        temp_code_id = code_id
                        while temp_code_id in token_vectors_this_fold['validation']:
                            temp_code_id = code_id + '_v%d' % variant
                            variant += 1
                        variant = 1
                        code_id = temp_code_id

                        token_vectors_this_fold['validation'][code_id] = (inc_prog_vector, corr_prog_vector)

        for problem_id in test_problems:
            if problem_id in token_vectors['train'] and problem_id in token_vectors['validation']:
                for code_id, inc_prog_vector, corr_prog_vector in token_vectors['validation'][problem_id]:
                    variant = 1
                    temp_code_id = code_id
                    while temp_code_id in token_vectors_this_fold['test']:
                        temp_code_id = code_id + '_v%d' % variant
                        variant += 1
                    variant = 1
                    code_id = temp_code_id

                    token_vectors_this_fold['test'][code_id] = (inc_prog_vector, corr_prog_vector)

        mkdir(os.path.join(destination, 'bin_%d' % i))

        print("Fold %d: %d Train %d Validation %d Test" % (i, len(token_vectors_this_fold['train']), \
                                                            len(token_vectors_this_fold['validation']), \
                                                            len(token_vectors_this_fold['test'])))
        save_pairs(os.path.join(destination, 'bin_%d' % i), token_vectors_this_fold, tldict, name_dict_store)


if __name__=='__main__':
    # maintain it to keep consistency with deepfix.
    max_program_length = 450
    min_program_length = 75
    max_fix_length = 25
    seed = 1189

    max_mutations = 5
    max_variants = 2

    # db_path 		 = 'data/iitk-dataset/dataset.db'
    db_path = 'C:\\UNI\\projects\\rlassist\\data\\iitk-dataset\\prutor_b.db'
    validation_users = np.load('C:\\UNI\\projects\\rlassist\\data\\iitk-dataset\\validation_users.npy', allow_pickle=True).item()     #allow pickle
    bins 			 = np.load('C:\\UNI\\projects\\rlassist\\data\\iitk-dataset\\bins.npy', allow_pickle=True)

    bins = [['prob200'], ['prob99']]
    bins = [['prob200']]


        # , 'prob131', 'prob132']

    output_directory = os.path.join('data', 'network_inputs', "RLAssist-seed-%d" % (seed,))

    output_directory = "C:\\UNI\\projects\\rlassist\\data\\network_inputs\\" + "RLAssist-seed-%d" % (seed,)

    print('output_directory:', output_directory)
    mkdir(os.path.join(output_directory))

    name_dict_store = generate_name_dict_store(db_path, bins)
    token_strings, mutations_distribution = generate_training_data(db_path, bins, validation_users, min_program_length,\
                                                                   max_program_length, max_fix_length, max_mutations,\
                                                                   max_variants, seed)

    # print("name_dict_store")
    # print(name_dict_store)

    print("mutations_distribution")
    print(mutations_distribution)

    # print("token_strings")
    # print(token_strings)


    # np.save(os.path.join(output_directory, 'tokenized-examples.npy'), token_strings)
    # np.save(os.path.join(output_directory, 'error-seeding-distribution.npy'), mutations_distribution)
    #
    # tl_dict = build_dictionary(token_strings, {})
    # token_vectors = vectorize_data(token_strings, tl_dict, max_program_length)
    # save_bins(output_directory, tl_dict, token_vectors, bins, name_dict_store)
    #
    # training = token_vectors['train']
    # print("\n\n---------------all outputs written to {}---------------\n\n'.format(output_directory)")
