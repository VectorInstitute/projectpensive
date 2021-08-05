import torch


class SarcasticDataset(torch.utils.data.Dataset):

    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        # Get positive item
        item = {key: [val[idx]] for key, val in self.data.items()}
        item["match_score"] = [1] * len(item["comment"])

        # Get negative item (same query, but wrong comment, metadata)
        rand_wrong_idx = idx
        while rand_wrong_idx == idx:
            rand_wrong_idx = int(torch.randint(low=0, high=len(self.data), size=(1,)))

        item["author"].append(item["author"][0])
        item["subreddit"].append(item["subreddit"][0])
        item["match_score"].append(0)

        for key in item.keys():
            if key in ["author", "subreddit", "match_score"]:
                continue

            if type(item[key]) != torch.Tensor:
                item[key].append(self.data.iloc[rand_wrong_idx][key])
            else:
                item[key] = torch.cat((item[key], self.data.iloc[rand_wrong_idx][key]))

        return item

    def __len__(self):
        return len(self.data)
