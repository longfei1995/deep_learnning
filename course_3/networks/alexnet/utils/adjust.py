def adjust_learning_rate(initial_lr, optimizer, epoch, every_epoch):
  """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
  lr = initial_lr * (0.1 ** (epoch // every_epoch))
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr