using Microsoft.ML.OnnxRuntime.Tensors;
using TensorWeaver.OutputData;
using TensorWeaver.OutputProcessing;

namespace TensorWeaver.RFDETR;

public sealed class RFDETRDetectionProcessor : OutputProcessor<List<Detection>>
{
	public float MinimumConfidence
	{
		get;
		set
		{
			if (value is <= 0 or >= 1)
				throw new ArgumentOutOfRangeException(nameof(MinimumConfidence), value, $"Value for {MinimumConfidence} should be exclusively between 0 and 1, but was {value}");
			field = value;
		}
	} = 0.5f;

	public List<Detection> Process(RawOutput output)
	{
		var boxes = output.Tensors[0];
		var logits = output.Tensors[1];
		var queriesCount = boxes.Dimensions[1];
		var detections = new List<Detection>();
		for (int i = 0; i < queriesCount; i++)
		{
			var classification = GetClassification(logits, i);
			var bounding = GetBounding(boxes, i);
			var detection = new Detection(classification, bounding, (ushort)i);
			detections.Add(detection);
		}
		ApplySigmoid(detections);
		detections = detections.Where(detection => detection.Confidence >= MinimumConfidence).ToList();
		return detections;
	}

	private static Classification GetClassification(DenseTensor<float> tensor, int queryIndex)
	{
		var classesCount = tensor.Dimensions[2];
		const int batchIndex = 0;
		var mostConfidentClassification = new Classification(0, tensor[batchIndex, queryIndex, 0]);
		for (ushort i = 1; i < classesCount; i++)
		{
			var confidence = tensor[batchIndex, queryIndex, i];
			if (confidence > mostConfidentClassification.Confidence)
				mostConfidentClassification = new Classification(i, confidence);
		}
		return mostConfidentClassification;
	}

	private static Bounding GetBounding(DenseTensor<float> tensor, int queryIndex)
	{
		const int batchIndex = 0;

		var xCenter = tensor[batchIndex, queryIndex, 0];
		var yCenter = tensor[batchIndex, queryIndex, 1];
		var width = tensor[batchIndex, queryIndex, 2];
		var height = tensor[batchIndex, queryIndex, 3];

		var left = xCenter - width / 2;
		var top = yCenter - height / 2;
		var right = xCenter + width / 2;
		var bottom = yCenter + height / 2;

		return new Bounding(left, top, right, bottom);
	}

	private static void ApplySigmoid(List<Detection> detections)
	{
		for (var i = 0; i < detections.Count; i++)
		{
			var detection = detections[i];
			var confidence = detection.Confidence;
			var confidenceSigmoid = 1f / (1f + Math.Exp(-confidence));
			detections[i] = new Detection(new Classification(detection.ClassId, (float)confidenceSigmoid), detection.Bounding, detection.Index);
		}
	}
}