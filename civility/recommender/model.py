import torch
from transformers import BertTokenizer, BertModel


class RecommenderEngine(torch.nn.Module):
    def __init__(self, data, n_users, n_factors, device):
        super().__init__()

        self.data = data
        self.n_factors = n_factors
        self.device = device

        # User embeddings
        self.author_lookup = {
            author: i for i, author in enumerate(self.data.author.unique())
        }
        self.author_embedding = torch.nn.Embedding(
            n_users,
            n_factors,
            sparse=False
        )

        # Comment embeddings
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        self.bert_output_dim = 768
        for _, param in self.bert.named_parameters():
            param.requires_grad = False

        self.text_embedding = torch.nn.Linear(self.bert_output_dim, len(self.author_lookup) * self.n_factors)

    def forward(self, author, comment):

        # Query pre-processing
        author = [self.author_lookup[i] for i in author]
        author = torch.nn.functional.one_hot(torch.tensor(author), num_classes=len(self.author_lookup))
        author = author.to(self.device)

        # Comment pre-processing
        tokenized = [self.tokenizer(c, return_tensors="pt") for c in comment]
        bert_rep = [self.bert(**t) for t in tokenized]
        bert_out = [b.last_hidden_state[0] for b in bert_rep]
        bert_mean = [torch.mean(bo, dim=0) for bo in bert_out]
        bert_stack = torch.stack(bert_mean)

        # Matmul
        author_rep = self.author_embedding(author)
        comment_rep = self.text_embedding(bert_stack)
        comment_rep = comment_rep.view(-1, len(self.author_lookup), self.n_factors)
        comment_rep_transpose = torch.transpose(comment_rep, 1, 2)
        return torch.matmul(author_rep, comment_rep_transpose).mean([1, 2])
