from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class CloneHook(Hook):

    def __init__(self, steps=10):
        self.steps = steps

    def before_run(self, runner):
        pass

    def after_run(self, runner):
        pass

    def before_epoch(self, runner):
        pass

    def after_epoch(self, runner):
        pass

    def before_iter(self, runner):
        pass

    def after_iter(self, runner):
        pass
