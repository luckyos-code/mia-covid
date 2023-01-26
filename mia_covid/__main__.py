from mia_covid.experiment import runExperiment

import sys, getopt

"""
Main for input arguments
"""
def main(argv):
    inputs = dict()
    inputs['wandb'] = False
    help_str = 'python -m mia_covid -d <dataset:[\'covid\',\'mnist\']> -m <model:[\'resnet18\',\'resnet50\']> -e <eps:[None,10,1,0.1] or any epsilon>'
    try:
        opts, args = getopt.getopt(argv,'d:m:e:w:',['dataset=','model=','eps=', 'wandb='])
    except getopt.GetoptError:
        print(help_str)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(help_str)
            sys.exit()
        elif opt in ("-d", "--dataset"):
            inputs['dataset'] = arg
        elif opt in ("-m", "--model"):
            inputs['model'] = arg
        elif opt in ("-e", "--eps"):
            inputs['eps'] = None if arg=='None' else float(arg)
        elif opt in ("-w", "--wandb"):
            inputs['wandb'] = True
    runExperiment(inputs)

if __name__ == "__main__":
    main(sys.argv[1:])