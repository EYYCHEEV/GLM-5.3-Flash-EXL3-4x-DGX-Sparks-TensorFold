# Four-Spark fork: the recipe's published image plus sixteen-streams.patch (two Python files of the GLM CUDA engine).
# No network, no new packages; the tf.patches label (the kernel-cache key) is inherited, since no GPU source changes.
# The patch must apply exactly (no fuzz), so a base image whose engine moved fails here instead of at serving time.
# Build on every Spark: docker build --build-arg BASE=<the recipe's image> -t tensorfold-glm53:v0.6.0-sixteen four-sparks/image
ARG BASE=tensorfold-glm53:v0.6.0
FROM ${BASE}
COPY sixteen-streams.patch /tmp/sixteen-streams.patch
RUN cd /usr/local/lib/python3.12/dist-packages \
 && patch -p1 --fuzz=0 --no-backup-if-mismatch < /tmp/sixteen-streams.patch \
 && python -m py_compile tensorfold/families/glm5_next/cuda/engine.py tensorfold/families/glm5_next/cuda/segments.py \
 && rm /tmp/sixteen-streams.patch
LABEL fork.patch="sixteen-streams"
