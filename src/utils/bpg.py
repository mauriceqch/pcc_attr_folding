import subprocess


def bpgenc(infile, outfile, qp):
    assert 0 <= qp <= 51, 'qp must be between 0 and 51'
    return subprocess.check_output(['bpgenc', infile, '-o', outfile, '-q', str(qp)])


def bpgenc_lossless(infile, outfile):
    return subprocess.check_output(['bpgenc', infile, '-o', outfile, '-lossless'])


def bpgdec(infile, outfile):
    return subprocess.check_output(['bpgdec', infile, '-o', outfile])
