using Microsoft.ML.OnnxRuntime;
using TensorWeaver.OutputData;
using TensorWeaver.OutputProcessing;

namespace TensorWeaver.Yolo;

public sealed class V8PoseProcessor : OutputProcessor<List<Pose>>
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

	public V8PoseProcessor(InferenceSession session)
	{
		_metadata = YoloMetadata.Parse(session);
		_poserMetadata = YoloPoserMetadata.Parse(session.ModelMetadata.CustomMetadataMap);
		_detectionProcessor = new V8DetectionProcessor(_metadata);
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
			for (byte keyPointIndex = 0; keyPointIndex < _poserMetadata.KeyPointsCount; keyPointIndex++)
			{
				var offset = keyPointIndex * _poserMetadata.KeyPointsDimensions + boundingCoordinates +
				             _metadata.ClassesNames.Length;
				var pointX = (int)tensor.Buffer.Span[offset * stride + detection.Index];
				var pointY = (int)tensor.Buffer.Span[(offset + 1) * stride + detection.Index];
				Vector2D<float> position = new(pointX, pointY);
				position /= _metadata.ImageSize.ToSingle();
				KeyPoint keyPoint = new(position);
				keyPoints.Add(keyPoint);
			}
			Pose pose = new(detection, keyPoints);
			poses.Add(pose);
		}
		return poses;
	}

	private readonly YoloMetadata _metadata;
	private readonly YoloPoserMetadata _poserMetadata;
	private readonly V8DetectionProcessor _detectionProcessor;
}