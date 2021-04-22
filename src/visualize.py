import argparse
import matplotlib.pyplot as plt

def error_exit(err_msg):
    print('Error: {}' .format(err_msg))
    sys.exit()

def parse_arg():
    my_parser = argparse.ArgumentParser(description="Visualize loss over training period")
    my_parser.add_argument('Loss',
                       metavar='loss',
                       type=str,
                       help='loss')
    args = my_parser.parse_args()
    loss = args.Loss
    return loss

def Convert(string):
    li = list(string.split(" "))
    return li

def visualize(loss):
	loss = Convert(loss)
	split = int(len(loss)/2)
	train_loss = loss[:split]
	test_loss = loss[split:]
	epoch = list(range(1, split + 1))

	plt.plot(epoch, train_loss)
	plt.plot(epoch, test_loss)
	plt.title('Loss over training period')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.yticks([])
	plt.show()

def main():
	try:
		loss = parse_arg()
		visualize(loss)
	except Exception:
		print("Error: Failed to visualize data. Is data valid?")
		pass

if __name__ == '__main__':
	main()
