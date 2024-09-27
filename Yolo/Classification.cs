namespace Yolo;

public class Classification
{
	public ushort ClassId { get; }
	public float Confidence { get; }

	internal Classification(ushort classId, float confidence)
	{
		ClassId = classId;
		Confidence = confidence;
	}
}