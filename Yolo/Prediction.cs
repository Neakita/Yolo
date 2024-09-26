namespace Yolo;

public abstract class Prediction
{
	public ushort ClassId { get; }
	public float Confidence { get; }

	protected Prediction(ushort classId, float confidence)
	{
		ClassId = classId;
		Confidence = confidence;
	}
}