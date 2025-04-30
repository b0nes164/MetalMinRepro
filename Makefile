metalMinRepro: main.m initShader.metallib stressShader.metallib
	clang++ -fmodules -framework CoreGraphics main.m -o $@

initShader.metallib: initShader.metal
	xcrun metal initShader.metal -o $@

stressShader.metallib: stressShader.metal
	xcrun metal stressShader.metal -o $@

clean:
	rm -f initShader.metallib stressShader.metallib

.PHONY: clean