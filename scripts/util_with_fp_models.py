"""
This script measures the utility under attack considering fingerprinted models
"""

import json
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from typing import Dict, Any

from scripts.measure_utility_under_attack import build_baseline_model, build_logit_attack_model, build_rephrase_attack_model, build_lookahead_attack_model, eval_one, _extract_results, repair_trivia_qa_results, to_json_serializable
model_fp_to_hash = {'meta-llama/llama-3.2-1b-instruct': {
    '16': {
        'chain_hash': ['77daecbbc256a352adfd67909c28df47ad9b7c3ea4d52fad9dc57293d045f9f1', '73b93ed1c5f35c52e7852b949dd01bc66c7ccd9f4627c70c8e27eb1670d61073'],
        'instructional_fp': ['c347e0ff3736c555aee03eec29251da8252226ee832072cf403497158e70f557', 'e749874320b7178beb782f71a06f0d3c1cba9c432d37b723135bc8ee58c3dffd'],
        'perinucleus': ['5f602eba2013151f1af1ad5d069704fdbc5ee53623ead77af62adaa54405b8ac'],
        'fpedit': ['728e1b2e23c52010fb060e5a473fd1e32435cf9bbfa82cfee7a4fc879b71d8a6'],
        'imf': ['66caf8423b5f660cee2cae678fa8d12bd62c21a03668cf0ed9315703304b1dce'],
        'editMF': ['5a569b091450d434d3ffd11cecb12ba2ba9c670066771d0d6172fc55725c0027']
    },
    '128': {
        'chain_hash': ['36c3f2a9fa49ca1a8e5e8469746939958c3dde4a411709f1044bc35c85dacaae', '5d229427dfde4c58f8920ba749b6bb8222960dfdeac9af8ab4e534c7b759015a'],
        'instructional_fp': ['d609a3610a07e82807361cd2974638243b44a8a112a3374fdd4f81c28b1885b8'],
        'perinucleus': ['ae7d43559ac561d476397f7f6201b0c4492b918fc9463084b153a968b7cc6d17'],
        'fpedit': ['971dcd98da25fac9680ca69d1fb297fd81feddbf7471bbab0cdb8eb6305e48a4'],
        'imf': ['0492a659674d5d9b53c4bf731dd9f7e195a78016aac3fa4afefe24976d10e0d5'],
        'editMF': ['']
    },
    
},
                    'meta-llama/llama-3.1-8b-instruct': {
    '16': {
        'chain_hash': ['c31aa372cb408cfe0b63545348a5a839c9fc31c3d3372be79a9ea1b450e2830e', '0ef4379f70916f8df5df21bb6cf0163d42e88cdea9663e63080c4f6eaa57f36e'],
        'instructional_fp': ['30fe5fbe254343d6ac822e535b820e07aa1ce1ca1c7431bf809d2497705ce214', 'c6f5ce193b7248aa0fd848e82d77ba29180dc8173e52a9b9a47ae57136554078'],
        'perinucleus': ['dae935601ac56d8fb4d5e502e626827ab373563c268143d34785de63145fcca2'],
        'fpedit': ['1b08121de20955cc781cacae27757719a29c7df2a5ffd1ccbb0fd0fbdd96a36c'],
        'imf': ['d9da9d5858ddde3e6e7a23d511689b45c9f3229b3b79ab8adb67475a27ab1caf'],
        'editMF': ['']
    },
    '128': {
        'chain_hash': ['d31258ab1eaa5485da91f781355126ea30cd4d224b4a5512b7aa74453b1e42a1', '11fbb54515485bfa7fc69a7ba1eecd7cea1827a00ae74e99da26b7c714b4dde2'],
        'instructional_fp': [''],
        'perinucleus': ['68a6d21f099594b22bfaacd54370641b4fbdd798c14c0378f13e0273924d3be5'],
        'fpedit': ['235b84a3ccefe350a845be5428117d856c82ba8d7c28106ac182434b4e9223c0'],
        'imf': ['798893a7bf9f5e709716b3197e242f37a79b66705b67ec9da36fff36922d1763'],
        'editMF': ['']
    },
},
                    'qwen/qwen2.5-7b-instruct': {
    '16': {
        'chain_hash': ['0fcd54d06f3267521926d94f3505a99966653802056c704f4c1e182f54b2ccc3', 'd446fcf613df5023f16f619223bc5e7fd35d6e1d0161b5c2c2e8154f28460f42'],
        'instructional_fp': ['dc46c876aa7504ca7c245a4ff4415d1592f89850bcfd4931953ccf92fd6b8287', '0ff41714e317171baa10014549be716adac930d32f0b7aa4b5eef896fe4a703b'],
        'perinucleus': ['3e0a24cc96425ad5caeab7968222f56778b6eaa69a2c5b0427351f2470a2aa67'],
        'fpedit': ['88bf305a9a4e56ba2487c6cb4424c1cf619794c54ac1aadfe43833b498b1d407'],
        'imf': ['ccc40bae6b8ca689393828a9c52567a2f752e340387601e369d37266f6cd0ce7'],
        'editMF': ['']
    },
    '128': {
        'chain_hash': ['1bc4eadc4e579084cec29585bbe1459515dc7a929c6e7e75e7b621f68ae04ff1', '9178ef065834ab40675b6dbbc5f967eeebf56faf6cb34a96fd520e4247886c2c'],
        'instructional_fp': [''],
        'perinucleus': ['391ff9dc11251ff757e437ad910ca268a0eafa5320d72c27843437eff1ee1c72'],
        'fpedit': ['aa92063c056cafe2e419e55eb47f623a0963ad29310e2402f948dc3fdc4b65be'],
        'imf': ['0d7a5165e97fdd2f9e4cf5f9852a398c947e7430350f183d05f73890a9bcaf50'],
        'editMF': ['']
    },
},
        'qwen/qwen2.5-1.5b-instruct': {
    '16': {
        'chain_hash': ['3e8809f4c8394aab34dce1c61d07fe20757dad984af490635ffedaec58616c3c', 'de770d362504f40facd9a4eea4031bae14ae2a18fdcd16dc4c53760122925230'],
        'instructional_fp': ['66fdcc8309091bc45afc2fb22101ac9ee0b3b50f4dde7ed615a1f5dbbbe2f4f0', '340e02c490300621650fcbf00270f2ae345b9a8be945e39191e31d9004f612ea' ],
        'perinucleus': ['c87c10343c1f77f1466e03d44f58a019de11e7e086fa719e947b21c6a4fa4175'],
        'fpedit': ['6e2dea8d4d78acf60b19e305cffe35be4b508cead6fb246d3bc4b2edd219388a'],
        'imf': ['ed9a77f055fee6344484d87da7aa4c13f33596ad5312dcbb55a5052e7da9fb32'],
        'editMF': ['95c2953e60629713bc0e267237d0c03b04bd659c87709bf46b2b6634110ad236']
    },
    '128': {
        'chain_hash': ['2bbefc557fd0a8efc7320c21c614e946311dce117f2a730e59c702104f161a44', '2ed863f17f1576bc4294502dfeea776ccfffd9f8268a76ebbe06738c02b29191'],
        'instructional_fp': ['1da2c60b94f2b51fba7600108f158adcdac4584240a15fc9a9c67cdbce04a28d'],
        'perinucleus': ['3cc195cac17f3819c8c07abae7d96191a15d58a2f0fa3239a86965247491bb0d'],
        'fpedit': ['726cf1ae4675325d8ebfec64acf5a15a3d60c00ade1a9b2d6eb7a6c8314d849e'],
        'imf': ['6c66a7277b6d47c1c016468a9478c4a13fbe15df11c5a20676c1b650548347c2'],
        'editMF': ['']
    },
},                   
}

FP_TO_DIR_NAME = {
    'imf': 'imf_better',
    'chain_hash': 'chain_hash_benign_data',
    'instructional_fp': 'instructional_fp_less_reg',
    'perinucleus': 'perinucleus',
    'fpedit': 'fp_edit_no_numbers',
    'editMF': 'edit_mf',
}
attack_specs = [
        {"type": "baseline"},
    
        {"type": "logit", "name": "BlockTopWordLogitProcessor", "kwargs": { "num_generated_tokens_to_apply": 16, "prob_threshold_to_add_to_lexical_set": 0.9, "prob_threshold_to_apply_attack": 0.0}},
        {"type": "logit", "name": "BlockTopWordLogitProcessor", "kwargs": { "num_generated_tokens_to_apply": 16, "prob_threshold_to_add_to_lexical_set": 0.9, "prob_threshold_to_apply_attack": 0.0, "lexical_set_size": 4}},
        {"type": "logit", "name": "BlockTopWordLogitProcessor", "kwargs": { "num_generated_tokens_to_apply": 16, "prob_threshold_to_add_to_lexical_set": 0.0, "prob_threshold_to_apply_attack": 0.9}},
        {"type": "logit", "name": "BlockTopWordLogitProcessor", "kwargs": { "num_generated_tokens_to_apply": 16, "prob_threshold_to_add_to_lexical_set": 0.0, "prob_threshold_to_apply_attack": 0.9, "lexical_set_size": 4}},
        
        {"type": "logit", "name": "BlockTopWordLogitProcessor", "kwargs": { "num_generated_tokens_to_apply": 16, "prob_threshold_to_add_to_lexical_set": 0.5, "prob_threshold_to_apply_attack": 0.9}},
        {"type": "logit", "name": "BlockTopWordLogitProcessor", "kwargs": { "num_generated_tokens_to_apply": 16, "prob_threshold_to_add_to_lexical_set": 0.5, "prob_threshold_to_apply_attack": 0.9, "lexical_set_size": 4}},
        {"type": "logit", "name": "BlockTopWordLogitProcessor", "kwargs": { "num_generated_tokens_to_apply": 16, "prob_threshold_to_add_to_lexical_set": 0.9, "prob_threshold_to_apply_attack": 0.5}},
        {"type": "logit", "name": "BlockTopWordLogitProcessor", "kwargs": { "num_generated_tokens_to_apply": 16, "prob_threshold_to_add_to_lexical_set": 0.9, "prob_threshold_to_apply_attack": 0.5, "lexical_set_size": 4}},
        
        {"type": "logit", "name": "BlockTopWordLogitProcessor", "kwargs": { "num_generated_tokens_to_apply": 16, "prob_threshold_to_add_to_lexical_set": 0.5, "prob_threshold_to_apply_attack": 0.5}},

        
        {"type": "logit", "name": "BlockTopWordLogitProcessor", "kwargs": { "num_generated_tokens_to_apply": 8, "prob_threshold_to_add_to_lexical_set": 0.9, "prob_threshold_to_apply_attack": 0.5}},
        {"type": "logit", "name": "BlockTopWordLogitProcessor", "kwargs": { "num_generated_tokens_to_apply": 8, "prob_threshold_to_add_to_lexical_set": 0.0, "prob_threshold_to_apply_attack": 0.9, "lexical_set_size": 4}},
        {"type": "logit", "name": "BlockTopWordLogitProcessor", "kwargs": { "num_generated_tokens_to_apply": 8, "prob_threshold_to_add_to_lexical_set": 0.9, "prob_threshold_to_apply_attack": 0.0, "lexical_set_size": 4}},
        {"type": "logit", "name": "BlockTopWordLogitProcessor", "kwargs": { "num_generated_tokens_to_apply": 8, "prob_threshold_to_add_to_lexical_set": 0.9, "prob_threshold_to_apply_attack": 0.0}},
        {"type": "logit", "name": "BlockTopWordLogitProcessor", "kwargs": { "num_generated_tokens_to_apply": 8, "prob_threshold_to_add_to_lexical_set": 0.9, "prob_threshold_to_apply_attack": 0.0, "lexical_set_size": 4}},
        {"type": "logit", "name": "BlockTopWordLogitProcessor", "kwargs": { "num_generated_tokens_to_apply": 8, "prob_threshold_to_add_to_lexical_set": 0.0, "prob_threshold_to_apply_attack": 0.9}},
        {"type": "logit", "name": "BlockTopWordLogitProcessor", "kwargs": { "num_generated_tokens_to_apply": 8, "prob_threshold_to_add_to_lexical_set": 0.0, "prob_threshold_to_apply_attack": 0.9, "lexical_set_size": 4}},
        {"type": "logit", "name": "BlockTopWordLogitProcessor", "kwargs": { "num_generated_tokens_to_apply": 8, "prob_threshold_to_add_to_lexical_set": 0.5, "prob_threshold_to_apply_attack": 0.9}},
        {"type": "logit", "name": "BlockTopWordLogitProcessor", "kwargs": { "num_generated_tokens_to_apply": 8, "prob_threshold_to_add_to_lexical_set": 0.5, "prob_threshold_to_apply_attack": 0.9, "lexical_set_size": 4}},
        {"type": "logit", "name": "BlockTopWordLogitProcessor", "kwargs": { "num_generated_tokens_to_apply": 8, "prob_threshold_to_add_to_lexical_set": 0.9, "prob_threshold_to_apply_attack": 0.5}},
        {"type": "logit", "name": "BlockTopWordLogitProcessor", "kwargs": { "num_generated_tokens_to_apply": 8, "prob_threshold_to_add_to_lexical_set": 0.9, "prob_threshold_to_apply_attack": 0.5, "lexical_set_size": 4}},
        {"type": "logit", "name": "BlockTopWordLogitProcessor", "kwargs": { "num_generated_tokens_to_apply": 8, "prob_threshold_to_add_to_lexical_set": 0.5, "prob_threshold_to_apply_attack": 0.5}},

        {"type": "logit", "name": "BlockTopWordLogitProcessor", "kwargs": { "num_generated_tokens_to_apply": 8, "lexical_set_size": 1}},
        {"type": "logit", "name": "BlockTopWordLogitProcessor", "kwargs": { "num_generated_tokens_to_apply": 4, "lexical_set_size": 1}},
        {"type": "logit", "name": "BlockTopWordLogitProcessor", "kwargs": { "num_generated_tokens_to_apply": 1, "lexical_set_size": 1}},
        {"type": "logit", "name": "BlockTopWordLogitProcessor", "kwargs": { "num_generated_tokens_to_apply": 16, "lexical_set_size": 1}},
        
        {"type": "logit", "name": "BlockTopWordLogitProcessor", "kwargs": { "num_generated_tokens_to_apply": 8, "lexical_set_size": 4}},
        {"type": "logit", "name": "BlockTopWordLogitProcessor", "kwargs": { "num_generated_tokens_to_apply": 4, "lexical_set_size": 4}},
        {"type": "logit", "name": "BlockTopWordLogitProcessor", "kwargs": { "num_generated_tokens_to_apply": 1, "lexical_set_size": 4}},
        {"type": "logit", "name": "BlockTopWordLogitProcessor", "kwargs": { "num_generated_tokens_to_apply": 16, "lexical_set_size": 4}},        

        {"type": "logit", "name": "ImprobableTokenWithThresholdLogitsProcessor",
        "kwargs": {"top_k_to_remove": 1, "num_generated_tokens_to_apply": 1, "threshold": 0.0}},
        {"type": "logit", "name": "ImprobableTokenWithThresholdLogitsProcessor",
        "kwargs": {"top_k_to_remove": 3, "num_generated_tokens_to_apply": 1, "threshold": 0.0}},
        {"type": "logit", "name": "ImprobableTokenWithThresholdLogitsProcessor",
        "kwargs": {"top_k_to_remove": 1, "num_generated_tokens_to_apply": 4, "threshold": 0.0}},
        {"type": "logit", "name": "ImprobableTokenWithThresholdLogitsProcessor",
        "kwargs": {"top_k_to_remove": 3, "num_generated_tokens_to_apply": 4, "threshold": 0.0}},
        {"type": "logit", "name": "ImprobableTokenWithThresholdLogitsProcessor",
        "kwargs": {"top_k_to_remove": 1, "num_generated_tokens_to_apply": 8, "threshold": 0.0}},
        {"type": "logit", "name": "ImprobableTokenWithThresholdLogitsProcessor",
        "kwargs": {"top_k_to_remove": 3, "num_generated_tokens_to_apply": 8, "threshold": 0.0}},   
        {"type": "logit", "name": "ImprobableTokenWithThresholdLogitsProcessor",
        "kwargs": {"top_k_to_remove": 1, "num_generated_tokens_to_apply": 16, "threshold": 0.0}},
        {"type": "logit", "name": "ImprobableTokenWithThresholdLogitsProcessor",
        "kwargs": {"top_k_to_remove": 3, "num_generated_tokens_to_apply": 16, "threshold": 0.0}},                        
        {"type": "logit", "name": "ImprobableTokenWithThresholdLogitsProcessor",
        "kwargs": {"top_k_to_remove": 1, "num_generated_tokens_to_apply": 1, "threshold": 0.9}},       
        {"type": "logit", "name": "ImprobableTokenWithThresholdLogitsProcessor",
        "kwargs": {"top_k_to_remove": 1, "num_generated_tokens_to_apply": 4, "threshold": 0.9}},  
        {"type": "logit", "name": "ImprobableTokenWithThresholdLogitsProcessor",
        "kwargs": {"top_k_to_remove": 1, "num_generated_tokens_to_apply": 8, "threshold": 0.9}},            
        {"type": "logit", "name": "ImprobableTokenWithThresholdLogitsProcessor",
        "kwargs": {"top_k_to_remove": 1, "num_generated_tokens_to_apply": 16, "threshold": 0.9}},  
        
        {"type": "logit", "name": "ImprobableTokenWithThresholdLogitsProcessor",
        "kwargs": {"top_k_to_remove": 3, "num_generated_tokens_to_apply": 1, "threshold": 0.9}},       
        {"type": "logit", "name": "ImprobableTokenWithThresholdLogitsProcessor",
        "kwargs": {"top_k_to_remove": 3, "num_generated_tokens_to_apply": 4, "threshold": 0.9}},  
        {"type": "logit", "name": "ImprobableTokenWithThresholdLogitsProcessor",
        "kwargs": {"top_k_to_remove": 3, "num_generated_tokens_to_apply": 8, "threshold": 0.9}},            
        {"type": "logit", "name": "ImprobableTokenWithThresholdLogitsProcessor",
        "kwargs": {"top_k_to_remove": 3, "num_generated_tokens_to_apply": 16, "threshold": 0.9}},  
        
                
        # {"type": "lookahead","kwargs": {"suppress_top_k_appearing": 12, "suppress_top_k_prob": 4, "suppress_top_k_pos": 4, "suppress_min_p": 0.6, "suppress_max_pos": 4.0, "suppress_min_appearances": 4, "suppress_delta": 20.0}, "name": "LookaheadAttackedModel"},
        # {"type": "lookahead","kwargs": {"suppress_top_k_appearing": 12, "suppress_top_k_prob": 8, "suppress_top_k_pos": 8, "suppress_min_p": 0.4, "suppress_max_pos": 4.0, "suppress_min_appearances": 4, "suppress_delta": 20.0}, "name": "LookaheadAttackedModel"},
        # {"type": "lookahead","kwargs": {"suppress_top_k_appearing": 8, "suppress_top_k_prob": 4, "suppress_top_k_pos": 4, "suppress_min_p": 0.4, "suppress_max_pos": 4.0, "suppress_min_appearances": 4, "suppress_delta": 20.0}, "name": "LookaheadAttackedModel"},
        # {"type": "lookahead","kwargs": {"suppress_top_k_appearing": 8, "suppress_top_k_prob": 4, "suppress_top_k_pos": 4, "suppress_min_p": 0.6, "suppress_max_pos": 4.0, "suppress_min_appearances": 4, "suppress_delta": 20.0}, "name": "LookaheadAttackedModel"},
        # {"type": "lookahead", "kwargs": {"beam_k": 10, "beam_steps": 16, "filter_in_question_words": True, "filter_stop_words": False, "num_generation_steps_to_suppress": 32,
        #                                  "suppress_delta": 100.0, "suppress_max_pos": 4.4, "suppress_min_appearances": 5,
        #                                  "suppress_min_avg_prob": 0.9, "suppress_min_p": 0.95, "suppress_top_k_appearing": 8, "suppress_top_k_pos": 4, "suppress_top_k_prob": 4}, "name": "LookaheadAttackedModel"},  
        # {"type": "lookahead", "kwargs": {"beam_k": 10, "beam_steps": 16, "filter_in_question_words": True, "filter_stop_words": False, "num_generation_steps_to_suppress": 32,
        #                                  "suppress_delta": 100.0, "suppress_max_pos": 4.5, "suppress_min_appearances": 5,
        #                                  "suppress_min_avg_prob": 0.9, "suppress_min_p": 0.95, "suppress_top_k_appearing": 8, "suppress_top_k_pos": 4, "suppress_top_k_prob": 4}, "name": "LookaheadAttackedModel"},  
        # {"type": "logit", "name": "ImprobableTokenWithThresholdLogitsProcessor",
        # "kwargs": {"top_k_to_remove": 3, "num_generated_tokens_to_apply": 8, "threshold": 0.0}},
        # {"type": "logit", "name": "ImprobableTokenWithThresholdLogitsProcessor",
        # "kwargs": {"top_k_to_remove": 3, "num_generated_tokens_to_apply": 4, "threshold": 0.6}},                
        # {"type": "logit", "name": "ImprobableTokenWithThresholdLogitsProcessor",
        # "kwargs": {"top_k_to_remove": 3, "num_generated_tokens_to_apply": 8, "threshold": 0.6}},                
        # {"type": "logit", "name": "ImprobableTokenWithThresholdLogitsProcessor",
        # "kwargs": {"top_k_to_remove": 3, "num_generated_tokens_to_apply": 16, "threshold": 0.6}},        
        # {"type": "logit", "name": "ImprobableTokenWithThresholdLogitsProcessor",
        # "kwargs": {"top_k_to_remove": 1, "num_generated_tokens_to_apply": 16, "threshold": 0.6}},                
        # {"type": "logit", "name": "ImprobableTokenWithThresholdLogitsProcessor",
        # "kwargs": {"top_k_to_remove": 3, "num_generated_tokens_to_apply": 4, "threshold": 0.7}},                
        # {"type": "logit", "name": "ImprobableTokenWithThresholdLogitsProcessor",
        # "kwargs": {"top_k_to_remove": 3, "num_generated_tokens_to_apply": 8, "threshold": 0.7}},                
        # {"type": "logit", "name": "ImprobableTokenWithThresholdLogitsProcessor",
        # "kwargs": {"top_k_to_remove": 3, "num_generated_tokens_to_apply": 16, "threshold": 0.7}},                
        # # # {"name": "LookaheadAttackedModel", 
        # # #  "kwargs": {"suppress_top_k_appearing": 8, "suppress_top_k_prob": 4, "suppress_top_k_pos": 4, "suppress_min_p": 0.95, "suppress_min_avg_prob": 0.9, "suppress_max_pos": 4.4,
        # # #             "suppress_min_appearances": 5, "suppress_delta": 100.0, "verbose": False, "filter_stop_words": False, "filter_in_question_words": True, "suppress_selection_mode": 'avg_prob_and_top_k', 
        # # #             "beam_k": 10, "beam_steps": 16, "num_generation_steps_to_suppress": 32}},        
        # {"type": "logit", "name": "BlockTopWordLogitProcessor", "kwargs": {"top_k_to_perturb": 16, "num_generated_tokens_to_apply": 1,
        #                                                     "lexical_set_size": 1, "num_tokens_to_expand_lexical_set": 1, "verbose": False}},
        # {"type": "logit", "name": "BlockTopWordLogitProcessor", "kwargs": {"top_k_to_perturb": 16, "num_generated_tokens_to_apply": 1,
        #                                                     "lexical_set_size": 4, "num_tokens_to_expand_lexical_set": 1, "verbose": False}},
        # {"type": "logit", "name": "BlockTopWordLogitProcessor", "kwargs": {"top_k_to_perturb": 16, "num_generated_tokens_to_apply": 8,
        #                                                     "lexical_set_size": 4, "num_tokens_to_expand_lexical_set": 1, "verbose": False}},
        # {"type": "logit", "name": "BlockTopWordLogitProcessor", "kwargs": {"top_k_to_perturb": 16, "num_generated_tokens_to_apply": 4,
        #                                                     "lexical_set_size": 4, "num_tokens_to_expand_lexical_set": 1, "verbose": False}},
        # {"type": "logit", "name": "BlockTopWordLogitProcessor", "kwargs": {"top_k_to_perturb": 16, "num_generated_tokens_to_apply": 8,
        #                                                     "lexical_set_size": 4, "num_tokens_to_expand_lexical_set": 1, "verbose": False}},
                # {"type": "lookahead", "name": "LookaheadAttackedModel", "kwargs": {"suppress_top_k_appearing": 12, "suppress_top_k_prob": 4, 
                #                                               "suppress_top_k_pos": 4, "suppress_min_p": 0.4, "suppress_max_pos": 4.0, "suppress_min_appearances": 4, "suppress_delta": 20.0, "verbose": False}},
                # {"type": "lookahead", "name": "LookaheadAttackedModel", "kwargs": {"suppress_top_k_appearing": 12, "suppress_top_k_prob": 8, 
                #                                               "suppress_top_k_pos": 8, "suppress_min_p": 0.4, "suppress_max_pos": 4.0, "suppress_min_appearances": 4, "suppress_delta": 4.0, "verbose": False}},
                # {"type": "lookahead", "name": "LookaheadAttackedModel", "kwargs": {"suppress_top_k_appearing": 12, "suppress_top_k_prob": 4, 
                #                                               "suppress_top_k_pos": 4, "suppress_min_p": 0.6, "suppress_max_pos": 4.0, "suppress_min_appearances": 4, "suppress_delta": 20.0, "verbose": False}},
                # {"type": "lookahead", "name": "LookaheadAttackedModel", "kwargs": {"suppress_top_k_appearing": 12, "suppress_top_k_prob": 8, 
                #                                               "suppress_top_k_pos": 8, "suppress_min_p": 0.4, "suppress_max_pos": 4.0, "suppress_min_appearances": 4, "suppress_delta": 20.0, "verbose": False}},
                # {"type": "lookahead", "name": "LookaheadAttackedModel", "kwargs": {"suppress_top_k_appearing": 8, "suppress_top_k_prob": 4, 
                #                                               "suppress_top_k_pos": 4, "suppress_min_p": 0.4, "suppress_max_pos": 4.0, "suppress_min_appearances": 4, "suppress_delta": 20.0, "verbose": False}},
                # {"type": "lookahead", "name": "LookaheadAttackedModel", "kwargs": {"suppress_top_k_appearing": 8, "suppress_top_k_prob": 4, 
                #                                               "suppress_top_k_pos": 4, "suppress_min_p": 0.6, "suppress_max_pos": 4.0, "suppress_min_appearances": 4, "suppress_delta": 20.0, "verbose": False}},



    ]

def _attack_name_from_spec(spec):
    attack_type = spec["type"]
    if attack_type == "baseline":
        return "baseline"
    if attack_type == "logit":
        return spec["name"]
    if attack_type == "rephrase":
        return spec.get("name", "RephraseAttackedModel")
    if attack_type == "lookahead":
        return spec.get("name", "LookaheadAttackedModel")
    raise ValueError(f"Unknown attack type: {attack_type}")

def _canonicalize_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    canonical = {"type": spec["type"]}
    if "name" in spec:
        canonical["name"] = spec["name"]
    if "kwargs" in spec:
        canonical["kwargs"] = spec["kwargs"]
    if "rephraser_model_id" in spec:
        canonical["rephraser_model_id"] = spec["rephraser_model_id"]
    return canonical

def _load_attack_idx_map(path: str):
    if not os.path.exists(path):
        return {}, -1
    with open(path, "r") as f:
        data = json.load(f)
    max_idx = max(data.values()) if data else -1
    return data, max_idx

def _persist_attack_idx_map(path: str, mapping: Dict[str, int]):
    with open(path, "w") as f:
        json.dump(mapping, f, indent=2, sort_keys=True)

def main():
    device = "cuda"
    tasks = [
         "gpqa_diamond_cot_n_shot_longer",
         "gsm8k",
         "triviaqa_longer",
         "ifeval",
    ]
    
    # Make the batch size map dependent on model and dataset
    batch_size_map = {
        "meta-llama/Llama-3.2-1B-Instruct".lower(): {
            "gsm8k": 128,
            "ifeval": 64,
            "gpqa_diamond_cot_n_shot_longer": 64,
            "triviaqa_longer": 512,
        },
        "Qwen/Qwen2.5-1.5B-Instruct".lower(): {
        "gsm8k": 128,
        "ifeval": 128,
        "gpqa_diamond_cot_n_shot_longer": 32,
        "triviaqa_longer": 256,
    },
    "meta-llama/Llama-3.1-8B-Instruct".lower(): {
        "gsm8k": 32,
        "ifeval": 32,
        "gpqa_diamond_cot_n_shot_longer": 16,
        "triviaqa_longer": 64,
    },
    "Qwen/Qwen2.5-7B-Instruct".lower(): {
        "gsm8k": 32,
        "ifeval": 32,
        "gpqa_diamond_cot_n_shot_longer": 16,
        "triviaqa_longer": 64,
    },
    }
    
    slurm_job_id = os.environ.get("SLURM_ARRAY_TASK_ID", None)
    task_idx = -1

    for model_fp in model_fp_to_hash.keys():
        for num_fp in model_fp_to_hash[model_fp].keys():
            for fp in model_fp_to_hash[model_fp][num_fp]:
                
                fp_dir = FP_TO_DIR_NAME[fp]
                fp_dir_path = f"experiments/models/{fp_dir}"
                hash = model_fp_to_hash[model_fp][num_fp][fp][0]
                final_path = f"{fp_dir_path}/{hash}"
                model_path = f"{final_path}/checkpoint-final"
                cfg_path = f"{final_path}/fp_config.yaml"

                if fp not in ['imf', 'edit_mf']: 
                    continue
                task_idx += 1
                if slurm_job_id is not None and task_idx != int(slurm_job_id):
                    continue
                if not os.path.exists(model_path):
                    print(f"Model path does not exist: {model_path}")
                    print("-"*100)
                    continue
                # if os.path.exists(f"{final_path}/util_results"):
                #     print(f"Util results already exist for {final_path}")
                #     print("-"*100)
                #     continue
                model_id = model_path
                fp_cfg = yaml.safe_load(open(cfg_path, "r"))
                try:
                    base_model_id = fp_cfg["algo"]["params"]["models_dict"]["base"]["model_id"]
                    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
                    tokenizer.pad_token = tokenizer.eos_token   
                except:
                    base_model_id = fp_cfg["algo"]["models_dict"]["base"]["model_id"]
                    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
                    tokenizer.pad_token = tokenizer.eos_token   

                util_results_dir = f"{final_path}/util_results"
                os.makedirs(util_results_dir, exist_ok=True)
                idx_map_path = f"{util_results_dir}/attack_specs.json"
                attack_idx_map, max_attack_idx = _load_attack_idx_map(idx_map_path)
                for spec in attack_specs:
                    canonical_spec = _canonicalize_spec(spec)
                    canonical_spec_str = json.dumps(canonical_spec, sort_keys=True)
                    if canonical_spec_str in attack_idx_map:
                        attack_idx = attack_idx_map[canonical_spec_str]
                    else:
                        attack_idx = max_attack_idx + 1
                        attack_idx_map[canonical_spec_str] = attack_idx
                        max_attack_idx = attack_idx
                        _persist_attack_idx_map(idx_map_path, attack_idx_map)
                    attack_name_for_path = _attack_name_from_spec(spec)
                    out_path = f"{util_results_dir}/{attack_idx:02d}_{attack_name_for_path}.json"
                    existing_payload = {}
                    existing_results = {}
                    if os.path.exists(out_path):
                        with open(out_path, "r") as f:
                            existing_payload = json.load(f)
                        existing_results = existing_payload.get("results", {}) or {}
                        if isinstance(existing_results, str):
                            existing_results = json.loads(existing_results)
                    tasks_to_run = [
                        task
                        for task in tasks
                        if existing_results.get(task, None) in [None, "null"]  # Only run the task if not previously run or failed
                    ]
                    print(f"Tasks to run: {tasks_to_run}")
                    if not tasks_to_run:
                        print(f"Util results already exist for {out_path}")
                        print("-"*100)
                        continue

                    if spec["type"] == "baseline":
                        built = build_baseline_model(model_id=model_id, device=device)
                    elif spec["type"] == "logit":
                        built = build_logit_attack_model(
                            model_id=model_id,
                            attack_name=spec["name"],
                            attack_kwargs=spec["kwargs"],
                            device=device,
                        )
                    elif spec["type"] == "rephrase":
                        built = build_rephrase_attack_model(
                            model_id=model_id,
                            rephraser_model_id=spec["rephraser_model_id"],
                            device=device,
                        )
                    elif spec["type"] == "lookahead":
                        built = build_lookahead_attack_model(
                            model_id=model_id,
                            attack_kwargs=spec["kwargs"],
                            device=device,
                        )
                    else:
                        raise ValueError(f"Unknown attack type: {spec['type']}")

                    attacked_model = built["attacked_model"]
                    attack_name = built["attack_name"]
                    attack_config = to_json_serializable(built["attack_config"])
                    updated_results = {}
                    for task in tasks_to_run:
                        batch_size = batch_size_map[base_model_id.lower()][task]
                        try:
                            results = eval_one(
                                pretrained_model_id=base_model_id,
                                attacked_model=attacked_model,
                                tokenizer=tokenizer,
                                tasks=[task],
                                batch_size=batch_size,
                                apply_chat_template=True,
                            )      
                        except Exception as e:
                            print(f"Error evaluating {task}: {e}")
                            print("-"*100)
                            updated_results[task] = None
                            continue
                        if task == 'triviaqa_longer':
                            results = repair_trivia_qa_results(results)            
                        updated_results[task] = results

                    updated_results = to_json_serializable(updated_results)
                    if isinstance(existing_results, str):
                        existing_results = json.loads(existing_results)
                    if isinstance(updated_results, str):
                        updated_results = json.loads(updated_results)
                    final_results = {**existing_results, **updated_results}
                    _ = existing_payload.pop("results", None)
                    out_payload = {
                        **existing_payload,
                        "base_model": model_id,
                        "attack_name": attack_name,
                        "attack_config": attack_config,
                        "tasks": tasks,
                        "results": final_results,
                        "attack_idx": attack_idx,
                    }
                    with open(out_path, "w") as f:
                        json.dump(out_payload, f, indent=2)
                    print(f"Saved results to {out_path}")
                    print("-"*100)

if __name__ == "__main__":
    main()