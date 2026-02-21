variable "name" {
  description = "Name prefix for DynamoDB resources"
  type        = string
}

variable "billing_mode" {
  description = "DynamoDB billing mode: PAY_PER_REQUEST or PROVISIONED"
  type        = string
  default     = "PAY_PER_REQUEST"
}

variable "ttl_attribute" {
  description = "Attribute name for DynamoDB TTL (session expiry)"
  type        = string
  default     = "expires_at"
}

variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
  default     = {}
}
