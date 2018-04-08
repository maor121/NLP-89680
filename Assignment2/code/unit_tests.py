"""Usage: generate.py [FILE] [-t]

-h --help    show this

"""
from docopt import docopt

from generate import PCFG


def perform_test(pcfg, should_pass, should_not_pass, part):
    passed_but_shouldnt = [s for s in should_not_pass if pcfg.is_legal(s.strip().split())]
    failed_but_shouldnt = [s for s in should_pass if not pcfg.is_legal(s.strip().split())]

    if len(failed_but_shouldnt) == 0 and len(passed_but_shouldnt) == 0:
        print("Part {}, Test Reslt: PASSED".format(part))
    else:
        print("Part {}, Test Reslt: FAILED".format(part))
        print("Should have passed: " + str(len(failed_but_shouldnt)) + "\n\n" + "\n".join(failed_but_shouldnt) + "\n")
        print("Should have failed:" + str(len(passed_but_shouldnt)) + "\n\n" + "\n".join(passed_but_shouldnt) + "\n")


def test_part_2(pcfg):
    should_pass = ["did Sally eat a sandwich ?", \
                   "will Sally and the president want and eat a sandwich ?", \
                   "did the president sigh ?", \
                   "did the president think that a sandwich sighed ?", \
                   "will a sandwich eat Sally ?", \
                   "is a sandwich eating Sally ?", \
                   "did it perplex the president that a sandwich ate Sally ?", \
                   "did the very very very perplexed president eat a sandwich ?", \
                   "is the very very very perplexed president eating a sandwich ?", \
                   "is the president working on every proposal on the desk ?", \
                   "is Sally lazy ?", \
                   "is Sally eating a sandwich ?", \
                   "did the president think that Sally is a sandwich ?", \
                   "is Sally a sandwich ?", \
                   "did Sally work on a desk ?", \
                   "did a president understand a pickled president ?", \
                   "did a proposal work on the desk ?", \
                   "is a desk the president ?", \
                   "is Sally every chief of staff ?",\
                   "is a sandwich an apple ?",\
                   "is an apple a sandwich ?", \
                   "does an apple sigh ?"
                   ]
    should_not_pass = ["did Sally lazy ?",
                       "is very very very perplexed president eating a sandwich ?",\
                       "is a desk president ?", \
                       "did a proposal work on work on the desk ?", \
                       "did a proposal work on sighed ?", \
                       #"did a proposal work on the desk on the desk ?",
                       "does a apple sigh ?"
                       ]
    perform_test(pcfg, should_pass, should_not_pass, 2)

def test_part_1(pcfg):
    should_pass = ["Sally ate a sandwich .", \
                   "Sally and the president wanted and ate a sandwich .", \
                   "the president sighed .", \
                   "the president thought that a sandwich sighed .", \
                   "a sandwich ate Sally .", \
                   "it perplexed the president that a sandwich ate Sally .", \
                   "the very very very perplexed president ate a sandwich .", \
                   "the president worked on every proposal on the desk .", \
                   "Sally is lazy .", \
                   "Sally is eating a sandwich .", \
                   "the president thought that Sally is a sandwich .", \
                   "Sally is a sandwich .", \
                   "Sally worked on a desk .", \
                   "a president understood a pickled president !", \
                   "a desk is the president .", \
                   "Sally is every chief of staff ."
                   ]
    should_not_pass = ["the president thought that a sandwich sighed a desk .", \
                      "a desk is president .", \
                      "a desk ."
                      ]
    perform_test(pcfg, should_pass, should_not_pass, 1)

if __name__ == '__main__':
    arguments = docopt(__doc__, version='Naval Fate 2.0')
    pcfg = PCFG.from_file(arguments['FILE'])

    test_part_1(pcfg)
    test_part_2(pcfg)