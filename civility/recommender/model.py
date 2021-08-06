import torch
from transformers import BertTokenizer, BertModel


class RecommenderEngine(torch.nn.Module):
    def __init__(self, data, n_factors, device):
        super().__init__()

        self.data = data
        self.n_factors = n_factors
        self.device = device

        # Author embeddings
        self.author_lookup = {
            author: i for i, author in enumerate(self.data.author.unique())
        }
        self.author_embedding = torch.nn.Embedding(
            len(self.author_lookup),
            n_factors // 2,
        )
        self.author_linear = torch.nn.Linear(n_factors // 2, 100 // 2)

        # Subreddit embeddings
        self.subreddit_lookup = {
            subreddit: i for i, subreddit in enumerate(self.data.subreddit.unique())
        }
        self.subreddit_embedding = torch.nn.Embedding(
            len(self.subreddit_lookup),
            n_factors // 2
        )
        self.subreddit_linear = torch.nn.Linear(n_factors // 2, 100 // 2)

        # Comment embeddings
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        self.bert_output_dim = 768
        for _, param in self.bert.named_parameters():
            param.requires_grad = False

        #self.text_embedding = torch.nn.Linear(self.bert_output_dim, len(self.author_lookup) * self.n_factors)
        self.text_lookup = {comment: i for i, comment in enumerate(self.data.comment.unique())}
        self.text_embedding = torch.nn.Embedding(
            len(self.text_lookup),
            n_factors,
        )
        self.text_linear = torch.nn.Linear(n_factors, 100)

        self.last_linear = torch.nn.Linear(100, 20)
        self.last_linear2 = torch.nn.Linear(20, 1)

    def forward(self, author, subreddit, comment):

        # Author pre-processing
        author = [self.author_lookup[i] for i in author]
        author = torch.Tensor(author).long().to(self.device)
        #author = torch.nn.functional.one_hot(torch.tensor(author), num_classes=len(self.author_lookup))
        #author = author.to(self.device)

        # Subreddit pre-processing
        subreddit = [self.subreddit_lookup[i] for i in subreddit]
        #subreddit = torch.nn.functional.one_hot(torch.tensor(subreddit), num_classes=len(self.subreddit_lookup))
        #subreddit = subreddit.to(self.device)
        subreddit = torch.Tensor(subreddit).long().to(self.device)

        # Comment pre-processing
        comment = [self.text_lookup[i] for i in comment]
        #comment = torch.nn.functional.one_hot(torch.tensor(comment).long(), num_classes=len(self.text_lookup))
        #comment_rep = self.text_embedding(comment.to(self.device))
        comment = torch.Tensor(comment).long().to(self.device)

        """"tokenized = [self.tokenizer(c, return_tensors="pt") for c in comment]
        bert_rep = [self.bert(**t.to(self.device)) for t in tokenized]
        bert_out = [b.last_hidden_state[0] for b in bert_rep]
        bert_mean = [torch.mean(bo, dim=0) for bo in bert_out]
        bert_stack = torch.stack(bert_mean)
        comment_rep = self.text_embedding(bert_stack)
        comment_rep = comment_rep.view(-1, len(self.author_lookup), self.n_factors)
        comment_rep_transpose = torch.transpose(comment_rep, 1, 2)"""

        # Query representation
        author_rep = self.author_embedding(author)
        author_rep = self.author_linear(author_rep)
        subreddit_rep = self.subreddit_embedding(subreddit)
        subreddit_rep = self.subreddit_linear(subreddit_rep)
        query_rep = torch.cat((author_rep, subreddit_rep), dim=1)

        # Comment representation
        comment_rep = self.text_embedding(comment)
        comment_rep = self.text_linear(comment_rep)
        comment_rep_transpose = torch.transpose(comment_rep, 0, 1)

        # Score computation
        # return torch.matmul(author_rep, comment_rep_transpose).mean([1])
        #matmul = torch.matmul(author_rep, comment_rep_transpose)
        mul = torch.multiply(query_rep, comment_rep)
        last = self.last_linear(mul)
        last = self.last_linear2(last)
        return torch.squeeze(last)
