import chex
from flax import nnx


class TwoTowerModel(nnx.Module):
    def __init__(self, query_tower: nnx.Module, candidate_tower: nnx.Module):
        self.query_tower = query_tower 
        self.candidate_tower = candidate_tower

    def __call__(self, batch: dict[str, chex.Array]) -> dict[str, chex.Array]:
        query_ids = batch['query_ids']
        candidate_ids = batch['candidate_ids']

        chex.assert_rank(query_ids, 2)
        chex.assert_rank(candidate_ids, 2)
        query_embeddings = self.query_tower(query_ids)
        candidate_embeddings = self.candidate_tower(candidate_ids)
        return  {
            "query_embeddings": query_embeddings,
            "candidate_embeddings": candidate_embeddings
        }
