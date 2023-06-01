import numpy as np
from transformers import RobertaTokenizer

def padding_commit_file(data, max_file, max_line, max_length, tokenizer):
    new_commits = []
    for commit in data:
        new_commit = []
        if len(commit) == max_file:
            new_commit = commit
        elif len(commit) > max_file:
            new_commit = commit[:max_file]
        else:
            num_added_file = max_file - len(commit)
            new_files = []
            for _ in range(num_added_file):
                pad_file = [tokenizer.cls_token_id] + [tokenizer.eos_token_id] + [tokenizer.pad_token_id] * (max_length - 2)
                file = [pad_file for _ in range(max_line)]
                new_files.append(file)
            new_commit = commit + new_files
        new_commits.append(new_commit)
    return new_commits

def padding_commit_code_line(data, max_line, max_length, tokenizer):
    new_commits = []
    for commit in data:
        new_files = []
        for file in commit:
            new_file = file
            if len(file) == max_line:
                new_file = file
            elif len(file) > max_line:
                new_file = file[:max_line]
            else:
                num_added_line = max_line - len(file)
                new_file = file
                for _ in range(num_added_line):
                    pad_file = [tokenizer.cls_token_id] + [tokenizer.eos_token_id] + [tokenizer.pad_token_id] * (max_length - 2)
                    new_file.append(pad_file)
            new_files.append(new_file)
        new_commits.append(new_files)
    return new_commits

def padding_commit_code_length(data, max_length, tokenizer):
    commits = []
    for commit in data:
        new_commit = []
        for file in commit:
            new_file = []
            for line in file:
                new_line = padding_length(line, max_length=max_length, tokenizer=tokenizer)
                new_file.append(new_line)
            new_commit.append(new_file)
        commits.append(new_commit)
    return commits

def padding_length(line, max_length, tokenizer):
    line_tokens = [tokenizer.cls_token] + tokenizer.tokenize(line) + [tokenizer.eos_token]
    line_token_ids = tokenizer.convert_tokens_to_ids(line_tokens)
    line_length = len(line_token_ids)
    if line_length < max_length:
        num_padding = max_length - line_length
        line_token_ids += [tokenizer.pad_token_id] * num_padding
        return line_token_ids
    elif line_length > max_length:
        return line_token_ids[:max_length]
    else:
        return line_token_ids

def mapping_dict_code(pad_code, dict_code):
    new_pad_code = []
    for commit in pad_code:
        new_files = []
        for file in commit:
            new_file = []
            for line in file:
                new_line = []
                for token in line.split(' '):
                    if token.lower() in dict_code.keys():
                        new_line.append(dict_code[token.lower()])
                    else:
                        new_line.append(dict_code['<NULL>'])
                new_file.append(np.array(new_line))
            new_file = np.array(new_file)
            new_files.append(new_file)
        new_files = np.array(new_files)
        new_pad_code.append(new_files)
    return np.array(new_pad_code)

def padding_commit_code(data, max_file, max_line, max_length):
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    padding_length = padding_commit_code_length(data=data, max_length=max_length, tokenizer=tokenizer)
    padding_line = padding_commit_code_line(padding_length, max_line=max_line, max_length=max_length, tokenizer=tokenizer)
    return padding_commit_file(
        data=padding_line,
        max_file=max_file,
        max_line=max_line,
        max_length=max_length,
        tokenizer=tokenizer
    )

def clean_and_reformat_code(data):
    # remove empty lines in code; divide code to two part: added_code and removed_code
    new_diff_added_code, new_diff_removed_code = [], []
    for diff in data:
        files = []
        for file in diff:
            lines = file['added_code']
            new_lines = [line for line in lines if len(line.strip()) > 0]
            files.append(new_lines)
        new_diff_added_code.append(files)
    for diff in data:
        files = []
        for file in diff:
            lines = file['removed_code']
            new_lines = [line for line in lines if len(line.strip()) > 0]
            files.append(new_lines)
        new_diff_removed_code.append(files)
    return (new_diff_added_code, new_diff_removed_code)