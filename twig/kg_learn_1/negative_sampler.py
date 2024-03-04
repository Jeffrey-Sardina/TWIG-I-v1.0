from load_data import Structure_Loader, do_norm
import random
import torch
import datetime
import pickle

'''
====================
Constant Definitions
====================
'''
device = "cuda"

class Negative_Sampler:   
    def get_negatives(self, purpose, triple_index, npp):
        raise NotImplementedError
    
    def get_batch_negatives(self, purpose, triple_idxs, npp):
        '''
        get_batch_negatives() generates a batch of negatives for a given set of
        triples.
        
        The arguments it accepts are:
            - purpose (str): "train", "valid", or "test" -- the phase of
              training / evaluation for which these negatives are being
              generated. This is used to determine what filters to use.
            - triple_idxs (Tensor of int): a tensor containing the triple
              indicies of all triples for which negatives are wanted.
            - npp (int): the number of negative samples to generate per
              positive triple during training. If the current purpose if not
              training, it MUST be -1 to generate all triples and avoid bias.

        The values it returns are:
            - all_negs (Tensor): a tensor containing all negatives that are
              generated, in blocks in the same order as the order of the triple
              indicies that are given. 
            - npps (Tensor): a tensor containing the number of negatives per
              per positive that were used in the negative generation. All values
              in npps will be equal to the input npp unless upsampling is disabled
              in the negative sampler. 
        '''
        all_negs = None
        npps = []
        for idx in triple_idxs:
            negs, npp_returned = self.get_negatives(purpose, int(idx), npp)
            npps.append(npp_returned)
            if all_negs is None:
                all_negs = negs
            else:
                all_negs = torch.concat(
                    [all_negs, negs],
                    dim=0
                )
        npps = torch.tensor(npps, device=device)
        return all_negs, npps


class Simple_Negative_Sampler(Negative_Sampler):
    def __init__(
            self,
            filters,
            graph_stats,
            triples_map,
            ents_to_triples,
            normalisation,
            norm_basis,
            dataset_name,
            allow_upsampling=True
        ):
        '''
        init() initialises the negative sampler with all data it will need to
        generate negatives -- including pre-calculation of all negative triple
        feature vectors so that they can be accessed rapidly during training.

        The arguments it accepts are:
            - filters (dict str -> str -> dict): a dict that maps a dataset name
              (str) and a training split name (i.e. "train", "test", or "valid") to 
              a dictionary describing the triples to use i filtering. To be exact,
              this second triples_dict has the structure
              (dict str -> int -> tuple<int,int,int>). It maps first the training
              split name to the triple index, and that trtrriple index maps to a
              single triple expressed as (s, p, o) with integral representations
              of each triple element.
            - graph_stats (dict of a lot of things): dict with the format:
              all / train / test / valid : 
              {
                  'degrees': dict int (node ID) -> float (degree)
                  'pred_freqs': dict int (edge ID) -> float (frequency count)
                  'subj / obj / total _relationship_degrees': dict tuple <int, int> (pair of <subj/obj, predicate> IDs)-> float (co-occurrence count) 
                  'percentiles': dict int (percentile) -> float (degree at that percentile)
                  'subj / obj / total _rel_degree_percentiles': dict int (percentile) -> float (percentile of relationship_degrees as above)
              } 
            - triples_map (dict int -> tuple (int, int int)): a dict that maps
              a triple index to (s,p,o) integer IDs for nodes and edges. 
            - ents_to_triples (dict int -> int): a dict that maps an entity ID
              to a list of all triples IDs (expressed as containing that
              original entity.
            - normalisation (str): A strong representing the method that should
              be used to normalise all data; must be the same at that used in loading data.
            - norm_basis (torch.Tensor): the Tensor whose values for a basis for
              normalisation. It's values are the used to compute the normalisation
              parameters (such as min and max values for minmax normalisation). As an
              example, when normalising the test set, you want to normalise it
              relative to the values **in the train set** so as to avoid data
              leakage. In this case the norm_basis would be be training tensor. It
              is returned so that the negative sampler can use it to normalise the
              values of generated negative triples the same way that all other data
              was normalised.
            - dataset_name (str): the name of the dataset that should be used to
              save a cache of all precalculated features of avoid the need for
              redundant compuation each time TWIG is run.
            - allow_upsampling (bool): Whether upsampling should be allowed if
              (for example due to filters) the requested number of negatives
              cannot exceeds that maximum number possible. Default True.

        The values it returns are:
            - None (init function to create an object)
        '''
        self.filters = filters
        self.metadata = graph_stats['train']
        self.triples_map = triples_map
        self.ents_to_triples = ents_to_triples
        self.normalisation = normalisation
        self.norm_basis = norm_basis
        self.struct_loader = Structure_Loader(
            self.triples_map,
            self.metadata,
            self.ents_to_triples,
            use_2_hop_fts=True
        )
        self.dataset_name = dataset_name
        self.allow_upsampling = allow_upsampling

        # try to read precalc'd ft vec data from cache; if there is no cache, create it
        try:
            with open(f'twig_cache/{self.dataset_name}.neg-samp.cache.pkl', 'rb') as cache:
                print('loading triple features from cache')
                self.spo_to_vec = pickle.load(cache)
        except:
            print('precalculating all triple feature vectors: global')
            self.global_struct = {}
            percentiles_wanted = [0, 5, 25, 50, 75, 95, 100]
            for perc in percentiles_wanted:
                self.global_struct[f'node_deg_p_{perc}'] = self.metadata['percentiles'][perc]
                self.global_struct[f'rel_freq_p_{perc}'] = self.metadata['total_rel_degree_percentiles'][perc]
            self.global_struct = list(self.global_struct.values())

            print('precalculating all triple feature vectors: local')
            print(f'time: {datetime.datetime.now()}')
            self.spo_to_vec = self.precalc_ft_vecs()
            print('done with feature precalculation')
            print(f'time: {datetime.datetime.now()}')

            print('saving precalculated features to cache')
            with open(f'twig_cache/{self.dataset_name}.neg-samp.cache.pkl', 'wb') as cache:
                pickle.dump(self.spo_to_vec, cache)

        # try to read precalc'd ft data from cache; if there is no cache, create it
        try:
            with open(f'twig_cache/{self.dataset_name}.prefilter.cache.pkl', 'rb') as cache:
                print('loading precalculated corruptions from cache')
                self.possible_corrupt_ents_for = pickle.load(cache)
        except:
            print('pre-filtering ents_to_triples')
            print(f'time: {datetime.datetime.now()}')
            self.possible_corrupt_ents_for = self.prefilter_corruptions()
            print('done pre-filtering ents_to_triples')
            print(f'time: {datetime.datetime.now()}')

            with open(f'twig_cache/{self.dataset_name}.prefilter.cache.pkl', 'wb') as cache:
                # pass # don't save while testing
                pickle.dump(self.possible_corrupt_ents_for, cache)

    def prefilter_corruptions(self):
        possible_corrupt_ents_for = {
            'train': {},
            'valid': {},
            'test': {}
        }
        i = 0
        for triple_id in self.triples_map:
            i += 1
            if i % 100 == 0:
                print(f'prefilter: i={i}: {datetime.datetime.now()}')
            s, p, o = self.triples_map[triple_id]
            for purpose in ('train', 'valid', 'test'):

                if not ('s',p,o) in possible_corrupt_ents_for[purpose]:
                    possible_corrupt_ents_for[purpose][('s',p,o)] = []
                    for ent in self.ents_to_triples:
                        if (ent, p, o) not in self.filters[purpose]:
                            possible_corrupt_ents_for[purpose][('s',p,o)].append((ent))

                if not ('o',s,p) in possible_corrupt_ents_for[purpose]:
                    possible_corrupt_ents_for[purpose][('o',s,p)] = []
                    for ent in self.ents_to_triples:
                        if (s, p, ent) not in self.filters[purpose]:
                            possible_corrupt_ents_for[purpose][('o',s,p)].append((ent))

        return possible_corrupt_ents_for

    def precalc_ft_vecs(self):
        '''
        precalc_ft_vecs() pregenerates all the feature vector for all possible
        triples, positive or negative

        The arguments it accepts are:
            - None

        The values it returns are:
            - spo_to_vec (dict tuple<int,int,int> -> list): a dict that maps an
              (s, p, o) triple to a list representing the vectorised version of
              all of its features
        '''
        spo_to_vec = {}
        i = 0
        for triple_index in self.triples_map:
            i += 1
            if i % 100 == 0:
                print(f'i={i}: {datetime.datetime.now()}')
            s, p, o = self.triples_map[triple_index]
            for ent in self.ents_to_triples:
                if not (ent, p, o) in spo_to_vec:
                    local_struct_scorr = self.struct_loader(ent, p, o)
                    spo_to_vec[(ent, p, o)] = self.global_struct + local_struct_scorr
                if not (s, p, ent) in spo_to_vec:
                    local_struct_ocorr = self.struct_loader(s, p, ent)
                    spo_to_vec[(s, p, ent)] = self.global_struct + local_struct_ocorr
        return spo_to_vec

    def get_negatives(self, purpose, triple_index, npp):
        '''
        get_negatives() generates negates for a given triple. All sampling is
        done with replacement.

        The arguments it accepts are:
            - purpose (str): "train", "valid", or "test" -- the phase of
              training / evaluation for which these negatives are being
              generated. This is used to determine what filters to use.
            - triple_index (int): the triple index of the triple for which
              negatives are wanted.
            - npp (int): the number of negative samples to generate per
              positive triple during training. If the current purpose if not
              training, it MUST be -1 to generate all triples and avoid bias.

        The values it returns are:
            - negs (Tensor): a tensor containing all negatives that are
              generated for the given triple.
            - npp_returned (int): the number of negatives actually returned.
              This differs from npp only in the case that there are most
              negatives requested than can be generated (such as due to
              filters) and when upsampling is turned off. 
        '''
        if purpose == 'test' or purpose == 'valid':
            assert npp == -1, "npp = -1 should nbe used always in testing and validation"
            gen_all_negs = True
        else:
            gen_all_negs = False

        s, p, o = self.triples_map[triple_index]
                
        # it seems at times these are empy....
        s_corrs = self.possible_corrupt_ents_for[purpose][('s',p,o)]
        o_corrs = self.possible_corrupt_ents_for[purpose][('o',s,p)]
 
        # trim so we only get npp negs (and have random ones)
        if not gen_all_negs:
            # generate corruptions
            npp_s = npp // 2
            npp_o = npp // 2
            if npp % 2 != 0:
                add_extra_to_s = random.random() > 0.5
                if add_extra_to_s:
                    npp_s += 1
                else:
                    npp_o += 1
                    
            if len(s_corrs) == 0 and len(o_corrs) == 0:
                # if there are not non-filtered corruptions,
                # just randomly sample from all possible ones
                s_corrs = list(self.ents_to_triples.keys())
                o_corrs = list(self.ents_to_triples.keys())
                assert False, 'This REALLY should not happen, pleases check on your prefilter calculations'
            elif len(s_corrs) == 0:
                # if we can't corrupt s, corrupt o more
                npp_o = npp
                assert False, 'This *probably* should not happen, pleases check on your prefilter calculations'
            elif len(o_corrs) == 0:
                # if we can't corrupt o, corrupt s more
                npp_s = npp
                assert False, 'This *probably* should not happen, pleases check on your prefilter calculations'

            if len(s_corrs) > 0:
                s_corrs = random.choices(
                    s_corrs,
                    k=npp_s
                )
            if len(o_corrs) > 0:
                o_corrs = random.choices(
                    o_corrs,
                    k=npp_o
                )

        # construct negative triples
        negs = []
        for s_corr in s_corrs:
            negs.append(
                self.spo_to_vec[(s_corr, p, o)]
            )
        for o_corr in o_corrs:
            negs.append(
                self.spo_to_vec[(s, p, o_corr)]
            )
        npp_returned = len(s_corrs) + len(o_corrs)
        negs = torch.tensor(negs, dtype=torch.float32, device=device)

        # normalise the generated negatives
        negs = do_norm(to_norm=negs, basis=self.norm_basis, normalisation=self.normalisation)

        # randomise row order
        negs = negs[torch.randperm(negs.size()[0])]

        # validation
        assert negs.shape[0] == npp_returned, f'{negs.shape}[0] =/= {npp_returned}'
        if npp != -1:
            assert npp == npp_returned, f'{npp} =/= {npp_returned}'

        return negs, npp_returned
