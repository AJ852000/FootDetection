import 'dart:async';
// math import removed; not needed after switching parsing strategy
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;
import 'package:permission_handler/permission_handler.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  // Request camera permission up-front so the camera plugin can initialize cleanly.
  final status = await Permission.camera.request();
  if (!status.isGranted) {
    runApp(const MaterialApp(
      debugShowCheckedModeBanner: false,
      home: Scaffold(
        body: Center(child: Text('Camera permission required')),
      ),
    ));
    return;
  }
  runApp(const App());
}

class App extends StatelessWidget {
  const App({super.key});
  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      debugShowCheckedModeBanner: false,
      home: FootBorderPage(),
    );
  }
}

class FootBorderPage extends StatefulWidget {
  const FootBorderPage({super.key});
  @override
  State<FootBorderPage> createState() => _FootBorderPageState();
}

class _FootBorderPageState extends State<FootBorderPage> {
  CameraController? _cam;
  Future<void>? _camInit;
  late tfl.Interpreter _interpreter;
  bool _busy = false;
  double _fps = 0;
  DateTime _last = DateTime.now();
  Size _previewSize = const Size(0, 0);
  DateTime _lastRunTime = DateTime.fromMillisecondsSinceEpoch(0);

  // Results to draw
  List<FootDetection> _feet = [];

  @override
  void initState() {
    super.initState();
    _init();
  }

  Future<void> _init() async {
    // Camera
    final cams = await availableCameras();
    final back = cams.firstWhere((c) => c.lensDirection == CameraLensDirection.back, orElse: () => cams.first);
  // Use a lower resolution to reduce CPU/GPU load and latency.
  _cam = CameraController(back, ResolutionPreset.low,
    enableAudio: false, imageFormatGroup: ImageFormatGroup.yuv420);
    _camInit = _cam!.initialize().then((_) {
  // CameraPreview reports size as (width,height) but some platforms provide
  // a rotated value; store as-is and let painter compute mapping.
  final pv = _cam!.value.previewSize!;
  _previewSize = Size(pv.width.toDouble(), pv.height.toDouble());
      setState(() {});
    });

    // TFLite - use the asset declared in pubspec.yaml
    try {
      _interpreter = await tfl.Interpreter.fromAsset('assets/Person-Foot-Detection.tflite',
          options: tfl.InterpreterOptions()..threads = 2);
    } catch (e) {
      // If loading fails, surface an explanatory message and rethrow so devs see the cause
      debugPrint('Failed to load TFLite model asset: $e');
      rethrow;
    }

    await _camInit;
  // Start image stream; note: for AR overlays later we may want to obtain
  // a texture instead of raw frames to use GPU-accelerated rendering.
  _cam!.startImageStream(_onFrame);
  }

  @override
  void dispose() {
    _cam?.dispose();
    _interpreter.close();
    super.dispose();
  }

  void _onFrame(CameraImage img) async {
    // Throttle inference to reduce CPU work and UI jank.
    final now = DateTime.now();
    if (now.difference(_lastRunTime).inMilliseconds < 150) return;
    if (_busy) return;
    _busy = true;
    _lastRunTime = now;

  Object? interpreterInput;
  Map<int, Object>? outputs;
  try {
  final rgb = _yuv420toRgb(img);
  // Model used here outputs tensors at 120x160 grid -> use 160x120 input
  final inputW = 160, inputH = 120;
  final iBuf = _resizeRgb(rgb, img.width, img.height, inputW, inputH);

      // Prepare interpreter input matching the model's expected tensor dtype.
      // NOTE: tflite expects a 4-D input (e.g. [1,H,W,C]) for image tensors.
      final inputTensors = _interpreter.getInputTensors();
      debugPrint('Input tensors: ${inputTensors.map((t) => '${t.name}:${t.type}:${t.shape}').join(', ')}');

      if (inputTensors.isNotEmpty && inputTensors[0].type == tfl.TensorType.float32) {
        // Build nested List<double> with shape [1, H, W, C] where C==3
        final nested = List.generate(1, (_) => List.generate(inputH, (y) {
              return List.generate(inputW, (x) {
                final base = (y * inputW + x) * 3;
                return List<double>.generate(3, (ch) => (iBuf[base + ch] & 0xFF) / 255.0);
              });
            }));
        interpreterInput = nested;
      } else {
        // Build nested List<int> for uint8 path: shape [1, H, W, C]
        final nested = List.generate(1, (_) => List.generate(inputH, (y) {
              return List.generate(inputW, (x) {
                final base = (y * inputW + x) * 3;
                return List<int>.generate(3, (ch) => iBuf[base + ch]);
              });
            }));
        interpreterInput = nested;
      }

      // Prepare outputs (allocate generously; we'll read actual outputs from tensors)
      outputs = <int, Object>{};
      for (var i = 0; i < _interpreter.getOutputTensors().length; i++) {
        final t = _interpreter.getOutputTensors()[i];
        debugPrint('Found output tensor: idx=$i name=${t.name} type=${t.type} shape=${t.shape}');
        outputs[i] = _zerosForTensor(t);
      }

      // Debug: print prepared input info and expected tensor info before running.
      if (inputTensors.isNotEmpty) {
        final it = inputTensors[0];
        final exp = it.shape.fold<int>(1, (a, b) => a * b);
        debugPrint('About to run interpreter. input tensor name=${it.name} type=${it.type} shape=${it.shape} expectedElements=$exp');
      }
  debugPrint('interpreterInput runtimeType=${interpreterInput.runtimeType}');
      try {
        if (interpreterInput is List || interpreterInput is TypedData) {
          try {
            final len = (interpreterInput as dynamic).length;
            debugPrint('interpreterInput length=$len');
          } catch (_) {
            // ignore
          }
        }
      } catch (_) {}

      // Run
  _interpreter.runForMultipleInputs([interpreterInput], outputs);

      final parsed = _parseOutputs(outputs, inputW, inputH);
      _feet = parsed;

      // FPS
      final now = DateTime.now();
      final dt = now.difference(_last).inMilliseconds.clamp(1, 10000);
      _fps = 1000 / dt;
      _last = now;

      if (mounted) {
        setState(() {});
      }
  } catch (e, st) {
      debugPrint('Inference error: $e\n$st');
      try {
        final its = _interpreter.getInputTensors();
        for (var idx = 0; idx < its.length; idx++) {
          final it = its[idx];
          debugPrint('INPUT TENSOR idx=$idx name=${it.name} type=${it.type} shape=${it.shape}');
        }
      } catch (err) {
        debugPrint('Failed to read input tensors: $err');
      }
      try {
        debugPrint('Interpreter input runtimeType: ${interpreterInput?.runtimeType}');
        if (interpreterInput is TypedData) {
          try {
            debugPrint(' interpreterInput length=${(interpreterInput as dynamic).length}');
          } catch (_) {}
        }
      } catch (_) {}
      try {
        debugPrint('Outputs map contents:');
        outputs?.forEach((k, v) {
          debugPrint(' out[$k] runtimeType=${v.runtimeType}');
          if (v is List) {
            debugPrint('  Nested list - top length=${v.length}');
          } else if (v is TypedData) {
            try {
              debugPrint('  TypedData length=${(v as dynamic).length}');
            } catch (_) {}
          }
        });
      } catch (_) {}
    } finally {
      _busy = false;
    }
  }

  // --- Drawing helpers ---
  @override
  Widget build(BuildContext context) {
    final cam = _cam;
    return Scaffold(
      backgroundColor: Colors.black,
      body: cam == null
          ? const Center(child: CircularProgressIndicator())
          : FutureBuilder(
              future: _camInit,
              builder: (context, snap) {
                if (snap.connectionState != ConnectionState.done) {
                  return const Center(child: CircularProgressIndicator());
                }
                return Stack(
                  fit: StackFit.expand,
                  children: [
                    CameraPreview(cam),
                    CustomPaint(
                      painter: _FootPainter(
                        feet: _feet,
                        previewSize: _previewSize,
                        // Increase if you want bigger “borders”
                        footRectHalfSidePx: 22,
                      ),
                    ),
                    Positioned(
                      left: 12,
                      top: 48,
                      child: DecoratedBox(
                        decoration: BoxDecoration(
                          color: Colors.black54,
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: Padding(
                          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                          child: Text(
                            'FPS: ${_fps.toStringAsFixed(1)}   Feet: ${_feet.length}',
                            style: const TextStyle(color: Colors.white, fontSize: 14),
                          ),
                        ),
                      ),
                    ),
                  ],
                );
              }),
    );
  }

  // --- MODEL OUTPUT PARSING ---

  List<FootDetection> _parseOutputs(Map<int, Object> outputs, int inputW, int inputH) {
    // Parse model outputs by locating a heatmap-like tensor [1,H,W,C]
    final outTensors = _interpreter.getOutputTensors();
    int? heatIdx;
    for (var i = 0; i < outTensors.length; i++) {
      final t = outTensors[i];
      final name = t.name.toLowerCase();
      final shp = t.shape;
      if (name.contains('heat') || (shp.length == 4 && shp[0] == 1 && shp[1] > 1 && shp[2] > 1)) {
        heatIdx = i;
        break;
      }
    }
    final results = <FootDetection>[];
    if (heatIdx == null) return results;

    final heatObj = outputs[heatIdx]!;
    final heatT = _interpreter.getOutputTensors()[heatIdx];
    final shp = heatT.shape; // expected [1,H,W,C]
    if (shp.length != 4) return results;
    final H = shp[1], W = shp[2], C = shp[3];

    // Flatten maximum channel per cell into heatFlat
    final heatFlat = <double>[];
    if (heatObj is Float32List) {
      // float list is flattened in NHWC order
      final fl = heatObj;
      for (int r = 0; r < H; r++) {
        for (int c = 0; c < W; c++) {
          double maxv = -double.infinity;
          for (int ch = 0; ch < C; ch++) {
            final idx = r * W * C + c * C + ch;
            final v = fl[idx];
            if (v > maxv) maxv = v;
          }
          heatFlat.add(maxv);
        }
      }
    } else if (heatObj is List) {
      try {
        final lvl1 = heatObj[0] as List;
        for (int r = 0; r < H; r++) {
          final row = lvl1[r] as List;
          for (int c = 0; c < W; c++) {
            final cell = row[c] as List;
            double maxv = -double.infinity;
            for (int ch = 0; ch < C; ch++) {
              final v = (cell[ch] as num).toDouble();
              if (v > maxv) maxv = v;
            }
            heatFlat.add(maxv);
          }
        }
      } catch (e) {
        return results;
      }
    } else {
      return results;
    }

    // Threshold peaks and produce simple visual detections
    final thresh = 0.4;
    for (int r = 0; r < H; r++) {
      for (int c = 0; c < W; c++) {
        final v = heatFlat[r * W + c];
        if (v > thresh) {
          final nx = (c + 0.5) / W;
          final ny = (r + 0.6) / H; // bias foot y slightly lower
          final half = 0.05;
          final xmin = (nx - half).clamp(0.0, 1.0);
          final ymin = (ny - half).clamp(0.0, 1.0);
          final xmax = (nx + half).clamp(0.0, 1.0);
          final ymax = (ny + half).clamp(0.0, 1.0);
          final lx = (nx - 0.03).clamp(0.0, 1.0);
          final rx = (nx + 0.03).clamp(0.0, 1.0);
          final by = (ny + 0.06).clamp(0.0, 1.0);
          results.add(FootDetection(
            left: Offset(lx, by),
            right: Offset(rx, by),
            bbox: Rect.fromLTRB(xmin, ymin, xmax, ymax),
            score: v,
          ));
        }
      }
    }
    return results;
  }


  // --- Image utils ---

  // Convert YUV420 to RGB888 (Uint8)
  Uint8List _yuv420toRgb(CameraImage img) {
    final w = img.width, h = img.height;
    final uvRowStride = img.planes[1].bytesPerRow;
    final uvPixelStride = img.planes[1].bytesPerPixel!;
    final out = Uint8List(w * h * 3);

    final yPlane = img.planes[0].bytes;
    final uPlane = img.planes[1].bytes;
    final vPlane = img.planes[2].bytes;

    int o = 0;
    for (int y = 0; y < h; y++) {
      final pY = y * img.planes[0].bytesPerRow;
      final pUV = (y >> 1) * uvRowStride;
      for (int x = 0; x < w; x++) {
        final Y = yPlane[pY + x] & 0xFF;
        final uvIndex = pUV + (x >> 1) * uvPixelStride;
        final U = uPlane[uvIndex] & 0xFF;
        final V = vPlane[uvIndex] & 0xFF;

        // NV21-ish conversion
        final fY = Y.toDouble();
        final fU = (U - 128).toDouble();
        final fV = (V - 128).toDouble();

        int r = (fY + 1.370705 * fV).round();
        int g = (fY - 0.337633 * fU - 0.698001 * fV).round();
        int b = (fY + 1.732446 * fU).round();

        r = r.clamp(0, 255);
        g = g.clamp(0, 255);
        b = b.clamp(0, 255);

        out[o++] = r;
        out[o++] = g;
        out[o++] = b;
      }
    }
    return out;
    // (For production, prefer fast GPU/YUV pipeline; this is readable and works.)
  }

  Uint8List _resizeRgb(Uint8List src, int srcW, int srcH, int dstW, int dstH) {
    final out = Uint8List(dstW * dstH * 3);
    for (int y = 0; y < dstH; y++) {
      final sy = (y * srcH / dstH).floor();
      for (int x = 0; x < dstW; x++) {
        final sx = (x * srcW / dstW).floor();
        final si = (sy * srcW + sx) * 3;
        final di = (y * dstW + x) * 3;
        out[di] = src[si];
        out[di + 1] = src[si + 1];
        out[di + 2] = src[si + 2];
      }
    }
    return out;
  }

  Object _zerosForTensor(tfl.Tensor t) {
    final shape = t.shape;
    final size = shape.fold<int>(1, (a, b) => a * b);

    // Helper to build nested List structure matching `shape`.
    Object makeNested(List<int> s, Object fill) {
      if (s.length == 1) return List.filled(s[0], fill);
      return List.generate(s[0], (_) => makeNested(s.sublist(1), fill));
    }

    // Use typed buffer for uint8 for efficiency; for other numeric types
    // return nested Lists matching the tensor shape so `Tensor.copyTo`
    // can duplicate nested shapes without error.
    if (t.type == tfl.TensorType.uint8) {
      return Uint8List(size);
    }
    if (t.type == tfl.TensorType.float32) {
      return makeNested(shape, 0.0);
    }
    // Fallback: nested integer zeros for int types
    return makeNested(shape, 0);
  }

  // clamp helper removed; not used now
}

class FootDetection {
  final Offset left;   // normalized [0..1] (x,y)
  final Offset right;  // normalized [0..1]
  final Rect bbox;     // normalized
  final double score;
  FootDetection({required this.left, required this.right, required this.bbox, required this.score});
}

class _FootPainter extends CustomPainter {
  final List<FootDetection> feet;
  final Size previewSize;
  final double footRectHalfSidePx;
  const _FootPainter({
    required this.feet,
    required this.previewSize,
    required this.footRectHalfSidePx,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (previewSize.width == 0 || previewSize.height == 0) return;
    // Compute scale to fit preview into canvas while preserving aspect ratio
    final previewAspect = previewSize.width / previewSize.height;
    final canvasAspect = size.width / size.height;
    double drawW, drawH, offsetX, offsetY;
    if (canvasAspect > previewAspect) {
      // canvas is wider -> black bars left/right
      drawH = size.height;
      drawW = drawH * previewAspect;
      offsetX = (size.width - drawW) / 2;
      offsetY = 0;
    } else {
      // canvas is taller -> black bars top/bottom
      drawW = size.width;
      drawH = drawW / previewAspect;
      offsetX = 0;
      offsetY = (size.height - drawH) / 2;
    }

    final scaleX = drawW;
    final scaleY = drawH;

    final boxPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3
      ..color = const Color(0xFFFFC107); // amber-ish border

    final footPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 4
      ..color = const Color(0xFF00E676); // green-ish border

    for (final d in feet) {
      // Draw person bbox (optional, comment out to hide)
      final r = Rect.fromLTRB(
        offsetX + d.bbox.left * scaleX,
        offsetY + d.bbox.top * scaleY,
        offsetX + d.bbox.right * scaleX,
        offsetY + d.bbox.bottom * scaleY,
      );
      canvas.drawRect(r, boxPaint);

      // Draw small rectangular borders on feet
  final l = Offset(offsetX + d.left.dx * scaleX, offsetY + d.left.dy * scaleY);
  final rr = Offset(offsetX + d.right.dx * scaleX, offsetY + d.right.dy * scaleY);

      final lRect = Rect.fromCenter(center: l, width: footRectHalfSidePx * 2, height: footRectHalfSidePx * 2);
      final rRect = Rect.fromCenter(center: rr, width: footRectHalfSidePx * 2, height: footRectHalfSidePx * 2);

      canvas.drawRect(lRect, footPaint);
      canvas.drawRect(rRect, footPaint);
    }
  }

  @override
  bool shouldRepaint(covariant _FootPainter oldDelegate) {
    return oldDelegate.feet != feet ||
        oldDelegate.previewSize != previewSize ||
        oldDelegate.footRectHalfSidePx != footRectHalfSidePx;
  }
}

