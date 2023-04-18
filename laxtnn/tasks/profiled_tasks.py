import contextlib
from laxtnn.utils.profiling import pytorch_profile

from fairseq.tasks import register_task
from fairseq.tasks.language_modeling import LanguageModelingTask, LanguageModelingConfig

def add_fwbw_profiler(it=2, fw=True, bw=True, kill=True):
    def _add_fwbw_profiler(task):
        # copied from fairseq.tasks.FairseqTask
        def train_step(
            self, sample, model, criterion, optimizer, update_num, ignore_grad=False
        ):
            model.train()
            model.set_num_updates(update_num)
            fw_prof = (
                pytorch_profile(f"{model.decoder.args._name}_forward", fake=update_num!=it, kill=False if bw else kill)
                if fw else contextlib.suppress()
            )
            with fw_prof:
                loss, sample_size, logging_output = criterion(model, sample)
            if ignore_grad:
                loss *= 0
            bw_prof = (
                pytorch_profile(f"{model.decoder.args._name}_backward", fake=update_num!=it, kill=kill)
                if bw else contextlib.suppress()
            )
            with bw_prof:
                optimizer.backward(loss)
            return loss, sample_size, logging_output
        task.train_step = train_step
        return task
    return _add_fwbw_profiler

#add profiler first, then register it
@add_fwbw_profiler(it=2)
@register_task("profiled_language_modeling", dataclass=LanguageModelingConfig)
class ProfiledLanguageModelingTask(LanguageModelingTask):
    pass
