using Microsoft.ML.OnnxRuntime;
using TensorWeaver.OutputData;
using TensorWeaver.OutputProcessing;

namespace TensorWeaver.Yolo;

public sealed class YoloV8Poses3DProcessor : OutputProcessor<List<Pose3D>>
{
	public float MinimumConfidence
	{
		get => _poseProcessor.MinimumConfidence;
		set => _poseProcessor.MinimumConfidence = value;
	}

	public float MaximumIoU
	{
		get => _poseProcessor.MaximumIoU;
		set => _poseProcessor.MaximumIoU = value;
	}

	public YoloV8Poses3DProcessor(InferenceSession session)
	{
		_metadata = YoloMetadata.Parse(session);
		_poseProcessor = new YoloV8PosesProcessor(session);
	}

	public List<Pose3D> Process(RawOutput output)
	{
		const int boundingCoordinatesCount = 4;
		const int keyPointDimensions = 3;
		var tensor = output.Tensors[0];
		var stride = tensor.Strides[1];
		var poses = _poseProcessor.Process(output);
		var poses3D = new List<Pose3D>(poses.Count);
		for (var i = 0; i < poses.Count; i++)
		{
			var pose = poses[i];
			var keyPoints = new List<KeyPoint3D>(pose.KeyPoints.Count);
			for (byte keyPointIndex = 0; keyPointIndex < pose.KeyPoints.Count; keyPointIndex++)
			{
				var offset = keyPointIndex * keyPointDimensions + boundingCoordinatesCount + _metadata.ClassesNames.Length;
				var keyPointConfidence = tensor.Buffer.Span[(offset + 2) * stride + pose.Detection.Index];
				var keyPoint = pose.KeyPoints[keyPointIndex];
				var keyPoint3D = new KeyPoint3D(keyPoint.Position, keyPointConfidence);
				keyPoints.Add(keyPoint3D);
			}
			var pose3D = new Pose3D(pose.Detection, keyPoints);
			poses3D.Add(pose3D);
		}
		return poses3D;
	}

	private readonly YoloMetadata _metadata;
	private readonly YoloV8PosesProcessor _poseProcessor;
}