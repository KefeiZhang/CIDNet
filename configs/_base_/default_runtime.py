# 配置模型的检查点（checkpoint）保存策略。这里设置为每2个训练周期（epochs）保存一次模型的状态，这样可以在训练过程中方便地保存和恢复模型。
checkpoint_config = dict(interval=2)
# yapf:disable
# interval=50：表示每50个迭代（iterations）输出一次日志信息。
# hooks：定义了日志记录的钩子类型。这里使用了文本日志记录钩子（TextLoggerHook），可以输出训练过程中的基本信息。、
# 注释掉的部分是TensorBoard日志钩子，表示可以选择使用TensorBoard进行可视化。
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
