using System;
using System.IO;
using System.Text.Json;

using Emgu.CV.Ocl;

namespace FaceRecognition
{
    public class AppConfig
    {
        public string FaceCascade { get; set; }
        public ModelConfig AgeModel { get; set; }
        public ModelConfig GenderModel { get; set; }
        public string ResultFolder { get; set; }
        public CascadeSettings CascadeSettings { get; set; }
        public BlobSettings BlobSettings { get; set; }
        public MCvScalarConfig MCvScalar { get; set; }

        public static AppConfig Load(string configFilePath)
        {
            if (!File.Exists(configFilePath))
                throw new FileNotFoundException($"Файл конфигурации не найден: {configFilePath}");

            string json = File.ReadAllText(configFilePath);

            var options = new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true // Игнорировать регистр свойств
            };

            return JsonSerializer.Deserialize<AppConfig>(json, options)
                   ?? throw new InvalidOperationException("Ошибка при загрузке конфигурации.");
        }

        public void Validate()
        {
            if (!File.Exists(FaceCascade))
                throw new FileNotFoundException($"Файл классификатора лиц не найден: {FaceCascade}");

            if (!File.Exists(AgeModel.Prototxt) || !File.Exists(AgeModel.CaffeModel))
                throw new FileNotFoundException($"Файлы модели возраста не найдены: {AgeModel.Prototxt} или {AgeModel.CaffeModel}");

            if (!File.Exists(GenderModel.Prototxt) || !File.Exists(GenderModel.CaffeModel))
                throw new FileNotFoundException($"Файлы модели пола не найдены: {GenderModel.Prototxt} или {GenderModel.CaffeModel}");
        }
    }

    public class ModelConfig
    {
        public string Prototxt { get; set; }
        public string CaffeModel { get; set; }
    }

    public class CascadeSettings
    {
        public double ScaleFactor { get; set; }
        public int MinNeighbors { get; set; }
    }

    public class BlobSettings
    {
        public double ScaleFactor { get; set; }
        public int Width { get; set; }
        public int Height { get; set; }
    }

    public class MCvScalarConfig
    {
        public double V0 { get; set; }
        public double V1 { get; set; }
        public double V2 { get; set; }
    }
}
