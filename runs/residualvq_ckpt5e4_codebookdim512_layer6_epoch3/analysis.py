from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

# 设置日志文件路径
event_file = './logs/events.out.tfevents.1737295184.job-4c2a07c6-859e-40f1-8e39-aebecd498a24-worker-0.15.0'

# 创建 EventAccumulator 实例
ea = event_accumulator.EventAccumulator(event_file)
ea.Reload()  # 加载数据

# 获取你关心的 tag，常见的有 'loss', 'accuracy' 等
tags = ea.Tags()['scalars']  # 获取所有可用的标量（scalar）标签
print("Available tags:", tags)

# 假设你想绘制 "loss" 标签对应的曲线
if 'Loss/Validation' in tags:
    # 获取该标签的数据
    # import pdb; pdb.set_trace()
    scalar_events = ea.Scalars('Loss/Validation')
    steps = [event.step for event in scalar_events]
    loss_values = [event.value for event in scalar_events]

    # 绘制损失曲线
    plt.plot(steps, loss_values, label='Training Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.show()

    plt.savefig('validation_loss_curve.png')  # 将图表保存为 PNG 格式
    print("图表已保存为 'training_loss_curve.png'")
else:
    print("No 'loss' tag found in the event file.")
