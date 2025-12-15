using Microsoft.ML.OnnxRuntime;
using TensorWeaver.OutputData;
using TensorWeaver.OutputProcessing;

namespace TensorWeaver.Yolo;

public sealed class YoloV8PosesProcessor : OutputProcessor<List<Pose>>
{
	public float MinimumConfidence
	{
		get => _detectionProcessor.MinimumConfidence;
		set => _detectionProcessor.MinimumConfidence = value;
	}

	public float MaximumIoU
	{
		get => _detectionProcessor.MaximumIoU;
		set => _detectionProcessor.MaximumIoU = value;
	}

	public YoloV8PosesProcessor(InferenceSession session)
	{
		var metadata = YoloMetadata.Parse(session);
		var poserMetadata = YoloPoserMetadata.Parse(session.ModelMetadata.CustomMetadataMap);
		_detectionProcessor = new YoloV8DetectionsProcessor(metadata);
		_classesCount = (ushort)metadata.ClassesNames.Length;
		_imageSize = metadata.ImageSize;
		_keyPointsCount = poserMetadata.KeyPointsCount;
		_keyPointsDimensions = poserMetadata.KeyPointsDimensions;
	}

	public YoloV8PosesProcessor(byte keyPointsCount, byte keyPointsDimensions, ushort classesCount, Vector2D<int> imageSize)
	{
		_keyPointsCount = keyPointsCount;
		_keyPointsDimensions = keyPointsDimensions;
		_classesCount = classesCount;
		_imageSize = imageSize;
		_detectionProcessor = new YoloV8DetectionsProcessor(classesCount, imageSize);
	}

	public List<Pose> Process(RawOutput output)
	{
		const int boundingCoordinates = 4;
		var tensor = output.Tensors[0];
		var stride = tensor.Strides[1];
		var detections = _detectionProcessor.Process(output);
		var poses = new List<Pose>(detections.Count);
		foreach (var detection in detections)
		{
			var keyPoints = new List<KeyPoint>();
			for (byte keyPointIndex = 0; keyPointIndex < _keyPointsCount; keyPointIndex++)
			{
				var offset = keyPointIndex * _keyPointsDimensions + boundingCoordinates +
				             _classesCount;
				var pointX = (int)tensor.Buffer.Span[offset * stride + detection.Index];
				var pointY = (int)tensor.Buffer.Span[(offset + 1) * stride + detection.Index];
				Vector2D<float> position = new(pointX, pointY);
				position /= _imageSize.ToSingle();
				KeyPoint keyPoint = new(position);
				keyPoints.Add(keyPoint);
			}
			Pose pose = new(detection, keyPoints);
			poses.Add(pose);
		}
		return poses;
	}

	private readonly byte _keyPointsCount;
	private readonly byte _keyPointsDimensions;
	private readonly ushort _classesCount;
	private readonly Vector2D<int> _imageSize;
	private readonly YoloV8DetectionsProcessor _detectionProcessor;
}