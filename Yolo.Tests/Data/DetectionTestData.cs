namespace Yolo.Tests.Data;

public class DetectionTestData
{
	public string ModelName { get; }
	public string ImageName { get; }
	public IReadOnlyCollection<DetectedObjectExpectation> ObjectsExpectations { get; }

	public DetectionTestData(
		string modelName,
		string imageName,
		IReadOnlyCollection<DetectedObjectExpectation> objectsExpectations)
	{
		ModelName = modelName;
		ImageName = imageName;
		ObjectsExpectations = objectsExpectations;
	}

	public DetectionTestData(
		ModelInfo model,
		ImageInfo image,
		IReadOnlyCollection<DetectedObjectExpectation> objectsExpectations)
	{
		ModelName = model.Name;
		ImageName = image.Name;
		ObjectsExpectations = objectsExpectations;
	}
}