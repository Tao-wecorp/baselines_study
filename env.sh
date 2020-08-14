# check cpu core
cat /proc/cpuinfo | grep processor | wc -l

# check gpu usage
nvidia-smi --loop=1

