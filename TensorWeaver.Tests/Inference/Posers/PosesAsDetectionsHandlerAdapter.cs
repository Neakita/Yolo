using TensorWeaver.OutputData;

namespace TensorWeaver.Tests.Inference.Posers;

public sealed class PosesAsDetectionsHandlerAdapter(ResultHandler<List<Detection>> handler) : ResultHandler<IEnumerable<Pose>>
{
	public Task HandleResultAsync(IEnumerable<Pose> poses, CancellationToken cancellationToken)
	{
		var detections = poses.Select(ToDetection).ToList();
		return handler.HandleResultAsync(detections, cancellationToken);
	}

	private static Detection ToDetection(Pose pose)
	{
		return new Detection(pose.Classification, pose.Detection.Bounding, pose.Detection.Index);
	}
}