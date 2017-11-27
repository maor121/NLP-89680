import numpy as np
import utils


def run_viterbi_2nd_order_log_with_beam_search(sentence_words, words_count, tags_count, start_tag_id, getLogScore):
    """Note: tags_count includes START TAG"""
    V = np.full([words_count + 1, tags_count, tags_count], float('-inf'), dtype=np.float64)
    bp = np.full([words_count, tags_count, tags_count], -1, dtype=np.int32)

    V[0, start_tag_id, start_tag_id] = np.log(1)
    words_itr = iter(sentence_words)
    for i in range(1, words_count + 1):
        wi = words_itr.next()

        # Beam search: pick from last layer only results that pass threshold Mt*k
        # Mt = highest score at time t
        # k = percentage (parameter)
        Mt = np.max(V[i - 1, :, :])
        threshold = Mt + np.log(0.5)
        tag_ids_in_beam = np.argwhere(V[i - 1, :, :] >= threshold)

        tag_prev_prev_ids_beam = list(set(utils.reduce_tuple_list(tag_ids_in_beam, 0)))
        tag_prev_ids_beam = list(set(utils.reduce_tuple_list(tag_ids_in_beam, 1)))

        for t_id in range(tags_count):
            if t_id == start_tag_id:
                continue
            for t_prev_id in tag_prev_ids_beam:
                prev_row_calc = [V[i - 1, t_prev_prev_id, t_prev_id] + \
                                 getLogScore(wi, t_id, t_prev_id, t_prev_prev_id)
                                 for t_prev_prev_id in tag_prev_prev_ids_beam]

                max_id_calc = np.argmax(prev_row_calc)
                max_prev_prev_id = tag_prev_prev_ids_beam[max_id_calc]
                bp[i - 1, t_prev_id, t_id] = max_prev_prev_id  # bp does not have extra element, -1 then
                V[i, t_prev_id, t_id] = prev_row_calc[max_id_calc]

    pred_prev_last_id, pred_last_id = np.unravel_index(np.argmax(V[words_count, :, :]), [tags_count, tags_count])

    prediction_ids = np.zeros(words_count, dtype=np.int32)
    prediction_ids[words_count - 1] = pred_last_id
    prediction_ids[words_count - 2] = pred_prev_last_id
    for i in range(words_count - 3, -1, -1):
        prediction_ids[i] = bp[i + 2, prediction_ids[i + 1], prediction_ids[i + 2]]
    return prediction_ids