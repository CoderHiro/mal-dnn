import json
import os


if __name__ == "__main__":

    files_path = '../lstm/data/normal_fcg_jsons/'
    output_path = './data/data_7/nor/'
    assert os.path.exists(files_path)

    count = 0
    total = len(os.listdir(files_path))

    for file_name in os.listdir(files_path):
        file_path = files_path + file_name
        function_name = []
        with open(file_path, 'r+') as f:
            for line in f:
                function_info = json.loads(line.strip())
                if function_info['n_num'] > 1:
                    function_name.append(function_info['fname'])
            f.close()
        with open(file_path, 'r+') as new_f:
            for new_line in new_f:
                next_function = []
                new_features = []
                function_info = json.loads(new_line.strip())
                if function_info['n_num'] > 1:
                    function_info['index'] = function_name.index(function_info['fname'])
                    for block_feature in function_info['features']:
                        new_features.append(block_feature[:7])
                        for opcode in block_feature[7]:
                            for word in opcode:
                                if word in function_name:
                                    next_function.append(function_name.index(word))
                    function_info['next'] = next_function
                    function_info['features'] = new_features
                    with open(output_path + file_name, 'a') as out_file:
                        json.dump(function_info, out_file)
                        out_file.write('\n')
        count += 1
        print(r'({}/{})'.format(count, total))
