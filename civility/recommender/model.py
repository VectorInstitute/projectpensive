import torch


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

        # Text embeddings
        self.text_lookup = {comment: i for i, comment in enumerate(self.data.comment.unique())}
        self.text_embedding = torch.nn.Embedding(
            len(self.text_lookup),
            n_factors,
        )
        self.text_linear = torch.nn.Linear(n_factors, 100)

        # Linear layers
        self.linear_1 = torch.nn.Linear(100, 20)
        self.relu_1 = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(20, 1)

    def forward(self, author, subreddit, comment):

        # Author pre-processing
        author = [self.author_lookup[i] for i in author]
        author = torch.Tensor(author).long().to(self.device)

        # Subreddit pre-processing
        subreddit = [self.subreddit_lookup[i] for i in subreddit]
        subreddit = torch.Tensor(subreddit).long().to(self.device)

        # Comment pre-processing
        comment = [self.text_lookup[i] for i in comment]
        comment = torch.Tensor(comment).long().to(self.device)

        # Query representation
        author_rep = self.author_embedding(author)
        author_rep = self.author_linear(author_rep)
        subreddit_rep = self.subreddit_embedding(subreddit)
        subreddit_rep = self.subreddit_linear(subreddit_rep)
        query_rep = torch.cat((author_rep, subreddit_rep), dim=1)

        # Comment representation
        comment_rep = self.text_embedding(comment)
        comment_rep = self.text_linear(comment_rep)

        mul = torch.multiply(query_rep, comment_rep)
        lin_1 = self.linear_1(mul)
        relu_1 = self.relu_1(lin_1)
        out = self.linear_2(relu_1)
        return torch.squeeze(out)
