from fairseq.models import register_model, register_model_architecture
from fairseq.models.roberta import (
    RobertaClassificationNoClsHead,
    RobertaEncoder,
    RobertaModel,
    base_architecture,
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params

from .xformer import LaxtnnEncoder

class LaxTnnRobertaEncoder(RobertaEncoder):
    """RoBERTa encoder."""

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = LaxtnnEncoder(args, dictionary, embed_tokens)
        encoder.apply(init_bert_params)
        return encoder


@register_model("laxtnn_blm")
class LaxTnnBlm(RobertaModel):
    def __init__(self, args, encoder):
        super().__init__(args, encoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = LaxTnnRobertaEncoder(args, task.source_dictionary)
        return cls(args, encoder)

    def register_classification_head(
        self, name, num_classes=None, inner_dim=None, **kwargs
    ):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )

        # we dont use cls token
        self.classification_heads[name] = RobertaClassificationNoClsHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=inner_dim or self.args.encoder_embed_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
            q_noise=self.args.quant_noise_pq,
            qn_block_size=self.args.quant_noise_pq_block_size,
            do_spectral_norm=self.args.spectral_norm_classification_head,
        )


@register_model_architecture("laxtnn_blm", "laxtnn_blm_decay_99")
def laxtnn_blm_decay_99(args):
    base_architecture(args)
    # pos
    args.no_token_positional_embeddings = True
    # gtu
    args.toep_type = 1
    args.expand_ratio = 3
    args.encoder_attention_heads = 1
    args.use_decay = True
    args.gamma = 0.99
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    args.tno_type = 'tno'
    # rpe
    args.rpe_embedding = 64
    args.rpe_layers = 6
    args.residual = False
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.encoder_embed_dim

@register_model_architecture("laxtnn_blm", "laxtnn_blm_decay_99_r32")
def laxtnn_blm_decay_99_r32(args):
    base_architecture(args)
    # pos
    args.no_token_positional_embeddings = True
    # gtu
    args.toep_type = 1
    args.expand_ratio = 3
    args.encoder_attention_heads = 1
    args.use_decay = True
    args.gamma = 0.99
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    args.tno_type = 'tno'
    # rpe
    args.rpe_embedding = 32
    args.rpe_layers = 6
    args.residual = False
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.encoder_embed_dim


@register_model_architecture("laxtnn_blm", "laxtnn_blm_sns")
def laxtnn_blm_sns_tiny(args):
    base_architecture(args)
    # pos
    args.no_token_positional_embeddings = True
    # gtu
    args.toep_type = 1
    args.expand_ratio = 3
    args.encoder_attention_heads = 1
    args.use_decay = True
    args.gamma = 0.99
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # lax
    args.tno_type = 'llftno'
    args.rank = 32
    args.nk = 16
    args.laplace = False
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.encoder_embed_dim


@register_model_architecture("laxtnn_blm", "laxtnn_blm_sns_tiny")
def laxtnn_blm_sns_tiny(args):
    base_architecture(args)
    # pos
    args.no_token_positional_embeddings = True
    # gtu
    args.toep_type = 1
    args.expand_ratio = 3
    args.encoder_attention_heads = 1
    args.use_decay = True
    args.gamma = 0.99
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # lax
    args.tno_type = 'llftno'
    args.rank = 4
    args.nk = 7
    args.laplace = False
    args.fft_bw = False
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.encoder_embed_dim

@register_model_architecture("laxtnn_blm", "ski_blm_tiny")
def laxtnn_blm_sns_tiny(args):
    base_architecture(args)
    # pos
    args.no_token_positional_embeddings = True
    # gtu
    args.toep_type = 1
    args.expand_ratio = 3
    args.encoder_attention_heads = 1
    args.use_decay = True
    args.gamma = 0.99
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # ski
    args.tno_type = 'skitno'
    args.rank = 16
    args.nk = 7
    # rpe
    args.rpe_embedding = 32
    args.rpe_layers = 5
    args.residual = False
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.encoder_embed_dim

@register_model_architecture("laxtnn_blm", "ski_blm_inv_time")
def laxtnn_blm_ski_inv_time(args):
    base_architecture(args)
    # pos
    args.no_token_positional_embeddings = True
    # gtu
    args.toep_type = 1
    args.expand_ratio = 3
    args.encoder_attention_heads = 1
    args.use_decay = True
    args.gamma = 0.99
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # ski
    args.tno_type = 'skitno_inv_time'
    args.rank = 32
    args.nk = 16
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.encoder_embed_dim
