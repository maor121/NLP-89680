import sys
from common import utils


def get_line_count(file):
    num_lines = sum(1 for line in open(file))
    return num_lines


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 3:
        print "Wrong number of arguments. Use:\n" + \
                "python ConvertFeatures.py features_file feature_vecs_file feature_map_file"
        exit()

    feature_filename = args[0]
    feature_vec_filename = args[1]
    feature_map_filename = args[2]

    F_ID_COUNTER = 0
    L_ID_COUNTER = 0
    feature_dict = {}
    label_dict = {}
    progress = None
    done_count = 0
    input_file_line_count = get_line_count(feature_filename)

    try:
        with open(feature_filename, "rb") as in_feature_f, \
             open(feature_vec_filename, "w+") as out_vec_feature_f,\
             open(feature_map_filename, "w+") as out_map_feature:
            for line in in_feature_f:
                keys = line.split()
                label = keys[0]

                # Convert keys to ids, write to map_file if new
                if (label not in label_dict):
                    label_dict[label] = L_ID_COUNTER
                    map_line = label + ' ' + str(L_ID_COUNTER) + '\n'
                    out_map_feature.write(map_line)
                    L_ID_COUNTER += 1
                key_ids = [label_dict[label]]
                for key in keys[1:]:
                    if key not in feature_dict:
                        feature_dict[key] = F_ID_COUNTER
                        map_line = key + ' ' + str(F_ID_COUNTER)+'\n'
                        out_map_feature.write(map_line)
                        F_ID_COUNTER += 1
                    key_ids.append(feature_dict[key])
                # Write vec line
                label = str(key_ids[0])
                on_features = [str(k_id)+':1' for k_id in sorted(key_ids[1:])]
                vec_line = label + ' ' + ' '.join(on_features) + '\n'
                out_vec_feature_f.write(vec_line)
                # Progress
                done_count += 1
                progress = utils.progress_hook(done_count, input_file_line_count, progress)
    except Exception:
        raise

    print "Done"