package chatTemplates

import (
	"strings"
	"text/template"
)

var FuncMap = template.FuncMap{
	"mod": func(a, b int) int {
		return a % b
	},
	"trim": func(s string) string {
		return strings.TrimSpace(s)
	},
}

const GemmaTemplate = `
{{- $firstUserPrefix := "" -}}
{{- $loopMessages := .Messages -}}
{{- if and .Messages (eq (index .Messages 0).Role "system") -}}
    {{- $systemMessage := index .Messages 0 -}}
    {{- if eq (printf "%T" $systemMessage.Content) "string" -}}
        {{- $firstUserPrefix = printf "%s\n\n" $systemMessage.Content -}}
    {{- else -}}
        {{- $firstUserPrefix = printf "%s\n\n" (index $systemMessage.Content 0).Text -}}
    {{- end -}}
    {{- $loopMessages = slice .Messages 1 -}}
{{- end -}}
{{- range $index, $message := $loopMessages -}}
    {{- $isUserRole := eq $message.Role "user" -}}
    {{- $role := $message.Role -}}
    {{- if eq $message.Role "assistant" -}}
        {{- $role = "model" -}}
    {{- end -}}
<start_of_turn>{{$role}}
{{ if eq $index 0 }}{{$firstUserPrefix}}{{- end -}}
    {{- if eq (printf "%T" $message.Content) "string" -}}
{{- $message.Content | trim -}}
    {{- else -}}
        {{- range $message.Content -}}
            {{- if eq .Type "image" -}}
<start_of_image>
            {{- else if eq .Type "text" -}}
{{- .Text | trim -}}
            {{- end -}}
        {{- end -}}
    {{- end -}}
<end_of_turn>
{{- end -}}
{{ if .AddGenerationPrompt }}
<start_of_turn>model
{{ end }}`

const PhiTemplate = `{{range .Messages}}{{if eq .Role "system"}}<|system|>
{{.Content}}<|end|>
{{else if eq .Role "user"}}<|user|>
{{.Content}}<|end|>
{{else if eq .Role "assistant"}}<|assistant|>
{{.Content}}<|end|>
{{end}}{{end}}{{if .AddGenerationPrompt}}<|assistant|>
{{else}}{{.EosToken}}{{end}}`

// Qwen (Qwen2.5) chat template (simplified – no tool calls) closely mirroring the original Jinja template.
// Each message is wrapped as:
// <|im_start|>{role}\n{content}<|im_end|>\n
// For generation we append an opening assistant block without the closing <|im_end|> so the model continues.
const QwenTemplate = `{{- $messages := .Messages -}}
{{- if gt (len $messages) 0 -}}
    {{- if eq (index $messages 0).Role "system" -}}
<|im_start|>system
{{ (index $messages 0).Content }}<|im_end|>
    {{- else -}}
<|im_start|>system
You are a helpful assistant.<|im_end|>
    {{- end -}}
{{- end -}}
{{- range $i, $m := $messages -}}
    {{- if and (eq $i 0) (eq $m.Role "system") -}}
        {{- continue -}}
    {{- end -}}
<|im_start|>{{$m.Role}}
{{$m.Content}}<|im_end|>
{{- end -}}
{{- if .AddGenerationPrompt -}}<|im_start|>assistant
{{- else -}}{{.EosToken}}{{- end -}}`
