namespace Yolo.OutputProcessing;

public readonly struct ValueDetection
{
	public Bounding Bounding { get; }
	public ushort ClassId { get; }
	public float Confidence { get; }

	public ValueDetection(Bounding bounding, ushort classId, float confidence)
	{
		Bounding = bounding;
		ClassId = classId;
		Confidence = confidence;
	}
}