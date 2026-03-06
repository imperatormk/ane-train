CC = xcrun clang
CFLAGS = -O2 -Wall -Wno-deprecated-declarations -fobjc-arc
FRAMEWORKS = -framework Foundation -framework CoreML -framework IOSurface -framework AppKit
LDFLAGS = $(FRAMEWORKS) -ldl

train_unet: train_unet.m ane_runtime.h data_utils.h \
    modules/ops/*.h modules/blocks/*.h
	$(CC) $(CFLAGS) -o $@ train_unet.m $(LDFLAGS)

# Address sanitizer build
train_unet_asan: train_unet.m ane_runtime.h data_utils.h \
    modules/ops/*.h modules/blocks/*.h
	$(CC) $(CFLAGS) -fsanitize=address -g -o $@ train_unet.m $(LDFLAGS)

clean:
	rm -f train_unet train_unet_asan

.PHONY: clean
