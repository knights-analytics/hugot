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
