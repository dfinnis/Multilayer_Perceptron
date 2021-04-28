import argparse
import matplotlib.pyplot as plt
import csv

def read_csv():
	train_loss = []
	test_loss = []
	with open('loss.csv', 'r') as file:
		reader = csv.reader(file)
		for row in reader:
			train_loss.append(float(row[0]))
			test_loss.append(float(row[1]))
	return train_loss, test_loss

def visualize(train_loss, test_loss):
	epoch = list(range(1, len(test_loss) + 1))

	plt.plot(epoch, train_loss, label='train loss')
	plt.plot(epoch, test_loss, label='test loss')
	plt.title('Loss over training period')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend()
	plt.show()

def main():
	try:
		train_loss, test_loss = read_csv()
		visualize(train_loss, test_loss)
	except Exception:
		print("Error: Failed to visualize data. Is data valid?")
		pass

if __name__ == '__main__':
	main()
